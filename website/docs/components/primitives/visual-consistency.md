# Visual Consistency Mixins

`visual_consistency.py` contains the reusable visual-metric backends for AutoPipeline. It is implemented in `src/autopipeline/components/primitives/visual_consistency.py`.

This file is the shared backend layer for:

- SSIM-based metrics
- LPIPS
- SAM-backed object extraction
- depth prediction

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Primitive Group</span></div>

## Included Classes

| Class | Main role | Typical consumers |
| --- | --- | --- |
| `SSIMMixin` | masked or unmasked structural similarity | `ssim-pipe`, `depth-anything-v2-pipe` |
| `LPIPSMixin` | perceptual similarity | `lpips-pipe` |
| `SAMSegmentationMixin` | bbox-prompted SAM2 extraction | `sam-pipe`, `clip-pipe`, `dino-v3-pipe` |
| `DepthAnythingv2Mixin` | depth-map prediction | `depth-anything-v2-pipe` |

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

## SSIMMixin

### Public Methods

| Method | Purpose |
| --- | --- |
| `compute(ref_img_tensor, edited_img_tensor, mask=None, win_size=7, win_sigma=1.5)` | Wrapper around the low-level SSIM implementation. |

### Input / Output Contract

```python
compute(
    ref_img_tensor: torch.Tensor,
    edited_img_tensor: torch.Tensor,
    mask=None,
    win_size=7,
    win_sigma=1.5,
) -> float
```

### Behavior

The method calls the local `ssim(...)` utility with:

- `data_range=255`
- `size_average=True`

If the returned tensor is `NaN`, it replaces it with `-1e8`.

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

## LPIPSMixin

### Class Signature

```python
LPIPSMixin(**kwargs)
```

### Constructor Parameters

| Key | Default | Meaning |
| --- | --- | --- |
| `device` | auto | Torch device for LPIPS. |
| `net` | `alex` | LPIPS backbone name. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `compute(ref_img_tensor, edited_img_tensor)` | Compute LPIPS after normalizing tensors to `[-1, 1]`. |

### Input / Output Contract

Inputs are expected as image tensors in pixel scale `[0, 255]`. The method converts them internally to the LPIPS input range and returns one float score.

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

## SAMSegmentationMixin

### Class Signature

```python
SAMSegmentationMixin(**kwargs)
```

### Constructor Parameters

| Key | Required | Meaning |
| --- | --- | --- |
| `model_cfg` | Yes in practice | SAM2 config path. |
| `model_path` | Yes in practice | SAM2 checkpoint path. |
| `device` | No | Torch device for inference. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `get_best_mask_in_bbox(image_np, bbox)` | Run SAM and choose the highest-scoring mask for one bbox. |
| `crop_and_isolate_subject(image_np, bbox, mask, bg_color=(255,255,255))` | Replace background, then crop to bbox. |
| `extract_object_by_coord(image_np, coord, bg_color=(255,255,255))` | End-to-end bbox-to-isolated-crop helper. |

### Input / Output Contract

#### `get_best_mask_in_bbox(...)`

- input:
  - RGB numpy image
  - bbox as numpy array
- output:
  - best SAM mask
  - `None` if image setup fails

#### `crop_and_isolate_subject(...)`

- input:
  - image array
  - bbox
  - binary mask
- output:
  - cropped isolated subject image

#### `extract_object_by_coord(...)`

- output:
  - cropped isolated object
  - `None` if mask extraction fails

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

## DepthAnythingv2Mixin

### Class Signature

```python
DepthAnythingv2Mixin(**kwargs)
```

### Constructor Parameters

| Key | Required | Meaning |
| --- | --- | --- |
| `model_path` | Yes in practice | Depth-estimation checkpoint path. |
| `device` | No | Torch device for inference. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `get_depth_map(image, resize_to_original=True)` | Predict a dense depth map from a PIL image. |

### Input / Output Contract

Returns a numpy depth map, optionally resized to the original image size.

## Minimal Config Examples

### SAM-backed module init

```yaml
init_config:
  sam:
    model_cfg: ${user_config.model_paths.sam_cfg}
    model_path: ${user_config.model_paths.sam_path}
```

### Depth-backed module init

```yaml
init_config:
  model_path: ${user_config.model_paths.depth_anything_v2_path}
```

### LPIPS-backed module init

```yaml
init_config:
  net: alex
```

## Failure Semantics

Important behavior across this file:

- `SSIMMixin` converts `NaN` to `-1e8`
- `SAMSegmentationMixin.get_best_mask_in_bbox(...)` returns `None` if `set_image(...)` fails
- `SAMSegmentationMixin.extract_object_by_coord(...)` returns `None` when mask extraction fails
- LPIPS and depth mixins do not add custom recovery logic around model loading or inference

## Extension Notes

- Use this file for reusable backend logic, not metric-specific scoring semantics.
- If several modules need the same model-loading or preprocessing change, make that change here.
- If only one metric formula changes, keep the edit in the module layer instead.

</div>
