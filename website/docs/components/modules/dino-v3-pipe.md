# DINOv3Pipe

`DINOv3Pipe` is registered as `dino-v3-pipe` and implemented in `src/autopipeline/components/modules/dino_pipe.py`.

Its public surface mirrors `CLIPPipe`, but the semantics are more structure-oriented. In practice this pipe is used for:

- patch-structure consistency
- object-level DINO CLS similarity after SAM isolation

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `dino-v3-pipe` |
| Class | `DINOv3Pipe` |
| Main mixins | `DINOv3Mixin`, `MaskProcessor` |
| Optional dependency | `SAMSegmentationMixin` via nested `sam` config |

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor

```python
DINOv3Pipe(**kwargs)
```

### Supported init kwargs

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `model_path` | Yes in practice | none | DINOv3 checkpoint path. The current mixin expects a valid string path. |
| `device` | No | auto | Torch device for DINO inference. |
| `sam` | No | not set | Nested SAM config used only for `sam_dino_cls_sim`. |

### Derived attributes

During initialization the mixin sets:

- `input_image_size`
- `patch_size`

based on `EMBED_MODEL_RESOLUTION`.

If `sam` is present, the constructor creates a `SAMSegmentationMixin` instance and stores it as `self.sam_block`.

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `_calc_self_sim_matrix(features)` | Normalize patch features and compute a self-similarity matrix. |
| `calc_structure_similarity(ref_image, edited_image, bool_mask)` | Compare reference and edited self-similarity matrices. |
| `_pad_image_to_target_size(cropped_image, bg_color)` | Square-pad and resize a subject crop to the DINO input size. |
| `calc_object_pad_cls_sim(ref_image, edited_image, coords, bg_color)` | Compare DINO CLS features for paired object crops. |
| `__call__(...)` | Dispatch between structure similarity and SAM-backed CLS similarity. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Call Signature

```python
DINOv3Pipe.__call__(
    ref_image: Image.Image,
    edited_image: Image.Image,
    coords: List[Tuple[int, int, int, int]] = None,
    mask_mode: str = None,
    metric: str = "dinov3_structure_similarity",
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
| `coords` | Metric-dependent | Boxes used for masking or object pairing. |
| `mask_mode` | Only for structure mode | Region polarity, normally derived from pipeline `scope`. |
| `metric` | Yes | `dinov3_structure_similarity` or `sam_dino_cls_sim`. |

### Extra runtime kwargs

| Key | Used by | Default | Meaning |
| --- | --- | --- | --- |
| `patch_mask_threshold` | structure mode | `0.1` | Threshold for patch selection after resizing the region mask. |
| `bg_color` | SAM-backed CLS mode | `(255, 255, 255)` | Fill color for isolated object crops. |

## Supported Metrics

| Metric | What it measures | Return type | Better direction |
| --- | --- | --- | --- |
| `dinov3_structure_similarity` | similarity of patch self-similarity structure | `float` | higher is better |
| `sam_dino_cls_sim` | DINO CLS similarity on SAM-isolated paired object crops | `float` or `None` | higher is better |

### `dinov3_structure_similarity`

For this metric the pipe:

1. builds a resized region mask
2. converts it to a patch mask
3. extracts DINO patch embeddings
4. normalizes patch embeddings
5. computes reference and edited self-similarity matrices
6. measures MSE between those matrices
7. converts the loss into a bounded score with `1 / (1 + loss * 100)`

### `sam_dino_cls_sim`

This metric uses the same coordinate pairing convention as `sam_clip_cls_sim`:

- first half of `coords` -> reference objects
- second half of `coords` -> edited objects

Each paired crop is segmented with SAM, padded to a square, encoded with DINO, and compared by cosine similarity.

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Return Value

The pipe returns a single float score.

Depending on the metric branch:

- structure mode returns `0.0` if no valid patches survive masking
- object mode returns `None` if paired crops cannot be produced

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Examples

### Structure-preservation usage

```yaml
metric_configs:
  dinov3_structure_similarity:
    pipe_name: dino-v3-pipe
    default_config: ${pipes_default.dino-v3-pipe}
    init_config:
    scope: edit_area
    runtime_params:
      patch_mask_threshold: 0.1
```

### Object-reference usage with SAM

```yaml
metric_configs:
  sam_dino_cls_sim:
    pipe_name: dino-v3-pipe
    default_config: ${pipes_default.dino-v3-pipe}
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

The important soft-failure paths are:

- empty patch mask in structure mode -> returns `0.0`
- odd number of `coords` in object mode -> returns `None`
- empty SAM crop -> returns `None`

The most important init-time precondition is:

- `model_path` must be a valid DINO checkpoint path

Unlike `CLIPMixin`, the current `DINOv3Mixin` does not safely default `model_path`.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Extend this pipe when the backbone remains DINO-like and the output semantics stay feature-based.
- Keep the patch-mask and coordinate conventions aligned with `MaskProcessor` and `ParserGrounderPipe`.
- If you introduce a new DINO metric, add it as a new branch inside `__call__` and document the score direction explicitly.

</div>

</div>
