# Human and Hair Segmenters

`human_segmenter.py` implements the segmentation primitives used by human-centric evaluation. The file lives at `src/autopipeline/components/primitives/human_segmenter.py`.

Both classes are registry-backed through `EXPERT_REGISTRY`.

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Primitive Group</span></div>

## Registry Entries

| Registry key | Class | Main output |
| --- | --- | --- |
| `human-segmenter` | `HumanSegmentationMixin` | whole-person binary mask |
| `hair-segmenter` | `HairSegmentationMixin` | hair-region binary mask |

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--expert">Expert</span></div>

## HumanSegmentationMixin

### Constructor

```python
HumanSegmentationMixin(**kwargs)
```

#### Supported init kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `model_selection` | `1` | MediaPipe selfie-segmentation model variant. |
| `threshold` | `0.5` | Binary threshold applied to the segmentation confidence map. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `get_mask(image)` | Produce a binary human mask from a PIL image. |

### Input / Output Contract

```python
get_mask(image: Image.Image) -> np.ndarray | None
```

Returns:

- a binary `uint8` mask with shape `(H, W)` on success
- `None` if the segmentation output contains no active pixels

Foreground pixels are set to `1`.

### Failure Semantics

Soft-failure case:

- empty segmentation mask -> returns `None`

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--expert">Expert</span></div>

## HairSegmentationMixin

### Constructor

```python
HairSegmentationMixin(**kwargs)
```

#### Supported init kwargs

| Key | Required | Meaning |
| --- | --- | --- |
| `model_path` | Yes | Local MediaPipe hair-segmentation asset path. |

The constructor raises a `ValueError` if `model_path` is missing or does not exist.

### Public Methods

| Method | Purpose |
| --- | --- |
| `_convert_to_mp_image(image_input)` | Convert PIL or numpy input into `mp.Image`. |
| `get_mask(image)` | Produce a binary hair mask from a PIL image. |

### Input / Output Contract

#### `_convert_to_mp_image(image_input)`

Accepted inputs:

- `PIL.Image.Image`
- `np.ndarray`

Returns:

- `mp.Image`

Raises:

- `TypeError` for unsupported input types

#### `get_mask(image)`

Returns:

- a binary `uint8` mask with shape `(H, W)` on success
- `None` if the segmenter does not return a `category_mask`

## Minimal Config Example

```yaml
expert_configs:
  human-segmenter:
    model_selection: 1
    threshold: 0.5

  hair-segmenter:
    model_path: ${user_config.model_paths.hair_segmentation_path}
```

## Failure Semantics

Important behavior to know:

- `HumanSegmentationMixin` soft-fails to `None`
- `HairSegmentationMixin` hard-fails at init time if the model asset is missing
- `HairSegmentationMixin.get_mask(...)` soft-fails to `None` when no mask is produced

## Extension Notes

- Keep reusable segmentation backends here.
- Do not duplicate segmentation model loading inside body or hair metric modules.
- If you switch segmentation providers, this file is the correct extension point.

</div>
