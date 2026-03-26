# MaskProcessor

`MaskProcessor` is the bbox-to-mask utility implemented in `src/autopipeline/components/primitives/mask_processor.py`.

It is one of the most important reusable helpers in the framework because multiple modules depend on the same coordinate convention and the same region polarity semantics.

<div class="doc-kind-row"><span class="doc-kind doc-kind--utility">Utility</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Class Role

`MaskProcessor` exists to convert:

- pixel-space bbox lists

into:

- full-resolution masks
- resized masks
- patch-level masks

in the tensor layout required by each downstream metric backend.

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `make_mask(...)` | Build a mask at the original image resolution. |
| `make_resized_mask(...)` | Rescale coordinates first, then build a mask at target resolution. |
| `create_patch_mask_from_mask_2d(...)` | Downsample a high-resolution binary mask into patch-space. |

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## `make_mask(...)`

```python
make_mask(
    image_h: int,
    image_w: int,
    coords: list,
    *,
    return_format: str,
    mode: str = None,
)
```

### Parameters

| Parameter | Required | Meaning |
| --- | --- | --- |
| `image_h` | Yes | Source image height. |
| `image_w` | Yes | Source image width. |
| `coords` | Yes when `mode` is not `None` | List of `(x1, y1, x2, y2)` boxes. |
| `return_format` | Yes | `2d_numpy`, `3d_numpy`, or `4d_tensor`. |
| `mode` | No | Region polarity. If `None`, the method returns `None`. |

### Supported return formats

| `return_format` | Output shape | Typical consumer |
| --- | --- | --- |
| `2d_numpy` | `(H, W)` | CLIP / DINO patch masking |
| `3d_numpy` | `(3, H, W)` | LPIPS |
| `4d_tensor` | `(1, 3, H, W)` boolean tensor | SSIM-style computation |

### Region polarity

The implementation treats `mode` as:

- `outer`
  initialize to `1`, fill bbox region with `0`
- anything else non-`None`
  initialize to `0`, fill bbox region with `1`

In practice the pipeline uses:

- `edit_area -> inner`
- `unedit_area -> outer`

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## `make_resized_mask(...)`

```python
make_resized_mask(
    image_h: int,
    image_w: int,
    coords: list,
    *,
    return_format: str,
    mode: str = "outer",
    target_h: int,
    target_w: int,
)
```

### What it does

This method:

1. rescales each bbox from original image coordinates to target resolution
2. delegates to `make_mask(...)`

It is the standard path for patch-based backbones that operate on resized inputs.

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## `create_patch_mask_from_mask_2d(...)`

```python
create_patch_mask_from_mask_2d(
    mask_2d: np.ndarray,
    patch_size: int,
    threshold: float = 0.5,
) -> np.ndarray
```

### Parameters

| Parameter | Meaning |
| --- | --- |
| `mask_2d` | High-resolution binary mask. |
| `patch_size` | Patch width and height for the target backbone. |
| `threshold` | Fraction of active pixels required to mark a patch as active. |

### Output

Returns a low-resolution patch mask with shape:

```text
(H // patch_size, W // patch_size)
```

and values `0` or `1`.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Integration Example

`MaskProcessor` is usually used implicitly inside modules, but the effective runtime contract looks like:

```yaml
metric_configs:
  emd:
    pipe_name: clip-pipe
    scope: unedit_area
    runtime_params:
      patch_mask_threshold: 0.1
```

At runtime:

- `scope` becomes `mask_mode`
- `coords` come from parser-grounder
- `MaskProcessor` converts them into the correct mask layout

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

Important behavior to know:

- `mode is None` -> returns `None`
- unsupported `return_format` -> raises `ValueError`
- `create_patch_mask_from_mask_2d(...)` assumes the mask height and width are divisible by `patch_size`

That last assumption is not explicitly validated in the current implementation.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- If you change coordinate semantics, update this file first and document the change everywhere else.
- Reuse this helper instead of hand-writing bbox masking in each module.
- Keep `inner` and `outer` semantics stable. They are effectively part of the framework contract.

</div>

</div>
