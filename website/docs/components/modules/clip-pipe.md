# CLIPPipe

`CLIPPipe` is registered as `clip-pipe` and implemented in `src/autopipeline/components/modules/clip_pipe.py`.

It provides two related but distinct CLIP-based metrics:

- patch-level semantic distance outside or inside an edited region
- object-level CLS similarity on SAM-isolated subject crops

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `clip-pipe` |
| Class | `CLIPPipe` |
| Main mixins | `CLIPMixin`, `MaskProcessor` |
| Optional dependency | `SAMSegmentationMixin` via nested `sam` config |

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor

```python
CLIPPipe(**kwargs)
```

### Supported init kwargs

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `model_path` | No | `openai/clip-vit-base-patch32` | CLIP vision checkpoint. |
| `device` | No | auto | Torch device for CLIP inference. |
| `sam` | No | not set | Nested SAM config used only for `sam_clip_cls_sim`. |

### Derived attributes

During initialization the pipe also derives:

- `img_input_size`
- `patch_size`

from `EMBED_MODEL_RESOLUTION`.

If `sam` is provided, the constructor creates:

```python
self.sam_block = SAMSegmentationMixin(**dict(kwargs["sam"]))
```

That means `sam_clip_cls_sim` requires a valid nested SAM config at init time.

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `calc_emd(ref_image, edited_image, bool_mask)` | Compute Earth Mover's Distance over CLIP patch features. |
| `_pad_image_to_target_size(cropped_image, bg_color)` | Square-pad and resize an isolated object crop to the CLIP input size. |
| `calc_object_pad_cls_sim(ref_image, edited_image, coords, bg_color)` | Compare CLIP CLS features on paired, SAM-isolated object crops. |
| `__call__(...)` | Dispatch between `emd` and `sam_clip_cls_sim`. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Call Signature

```python
CLIPPipe.__call__(
    ref_image: Image.Image,
    edited_image: Image.Image,
    coords: List[Tuple[int, int, int, int]] = None,
    mask_mode: str = None,
    metric: str = "emd",
    **kwargs,
)
```

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Runtime Inputs

| Argument | Required | Meaning |
| --- | --- | --- |
| `ref_image` | Yes | Reference image. |
| `edited_image` | Yes | Edited image. |
| `coords` | Metric-dependent | Region boxes used for masking or object pairing. |
| `mask_mode` | Only for `emd` | `inner` or `outer`, usually derived from pipeline `scope`. |
| `metric` | Yes | `emd` or `sam_clip_cls_sim`. |

### Extra runtime kwargs

| Key | Used by | Default | Meaning |
| --- | --- | --- | --- |
| `patch_mask_threshold` | `emd` | `0.1` | Threshold when converting resized masks into patch masks. |
| `bg_color` | `sam_clip_cls_sim` | `(255, 255, 255)` | Fill color for isolated object crops. |

## Supported Metrics

| Metric | What it measures | Return type | Better direction |
| --- | --- | --- | --- |
| `emd` | Patch-level CLIP feature distance | `float` or `None` | lower is better |
| `sam_clip_cls_sim` | CLS cosine similarity on SAM-isolated object crops | `float` or `None` | higher is better |

### `emd`

For `emd`, the pipe:

1. resizes the region mask to the CLIP input resolution
2. converts the mask into a patch-level selection map
3. extracts reference features from the whole reference image
4. extracts edited features from only the selected edited patches
5. computes EMD with `ot.emd2(...)`

This metric is typically used with `scope: unedit_area`.

### `sam_clip_cls_sim`

For `sam_clip_cls_sim`, the pipe:

1. splits `coords` into two equal halves
2. treats the first half as reference boxes
3. treats the second half as edited-image boxes
4. uses SAM to isolate each object crop
5. pads each crop to a square
6. compares CLIP CLS embeddings

The coordinate list therefore must have even length and preserve reference/edit pairing order.

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Return Value

The pipe returns a single `float` score on success.

It returns `None` when the required region information cannot produce a valid comparison.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Examples

### Patch-distance usage

```yaml
metric_configs:
  emd:
    pipe_name: clip-pipe
    default_config: ${pipes_default.clip-pipe}
    init_config:
    scope: unedit_area
    runtime_params:
      patch_mask_threshold: 0.1
```

### Object-pair usage with SAM

```yaml
metric_configs:
  sam_clip_cls_sim:
    pipe_name: clip-pipe
    default_config: ${pipes_default.clip-pipe}
    init_config:
      sam: ${pipes_default.sam-pipe}
    scope: edit_area
    runtime_params:
      bg_color: !tuple [255, 255, 255]
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Modes

The most important soft-failure paths are:

- no valid patch selected after mask-to-patch conversion -> returns `None`
- `coords` length is odd for object pairing -> returns `None`
- SAM crop is empty for any paired object -> returns `None`

Two practical preconditions are not guarded by a custom error path:

- `sam_clip_cls_sim` assumes `self.sam_block` exists
- `coords` must follow the reference-half then edited-half ordering

If either assumption is violated, extension code should fix the config rather than patching downstream results.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Extend this pipe when the feature backbone remains CLIP-based.
- Add a new metric branch if you want a new way to compare CLIP features.
- Reuse `MaskProcessor` conventions so `scope` continues to map cleanly into `inner` and `outer`.
- If you need a different object isolator than SAM, make that change explicit in the constructor and config surface.

</div>

</div>
