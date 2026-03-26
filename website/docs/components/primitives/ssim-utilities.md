# SSIM Utilities

`ssim.py` contains the low-level structural-similarity implementation used by `SSIMMixin`. It is implemented in `src/autopipeline/components/primitives/ssim.py`.

This file is internal infrastructure, but it is still important for extension work because region-aware masking and tensor-shape rules are defined here.

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Primitive Group</span></div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--function">Function API</span></div>

## Top-Level Functions

| Function | Purpose |
| --- | --- |
| `_fspecial_gauss_1d(size, sigma)` | Build the Gaussian kernel used by SSIM. |
| `gaussian_filter(input, win)` | Apply separable Gaussian blur. |
| `_ssim(...)` | Internal SSIM core returning SSIM and contrast-structure terms. |
| `ssim(...)` | Public SSIM function with optional masking. |
| `ms_ssim(...)` | Multi-scale SSIM implementation. |

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Module Classes</span></div>

## Module Classes

| Class | Purpose |
| --- | --- |
| `SSIM` | `torch.nn.Module` wrapper around `ssim(...)` |
| `MS_SSIM` | `torch.nn.Module` wrapper around `ms_ssim(...)` |

## `ssim(...)`

```python
ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K=(0.01, 0.03),
    nonnegative_ssim: bool = False,
    mask: Optional[Tensor] = None,
) -> Tensor
```

### Input contract

| Parameter | Meaning |
| --- | --- |
| `X`, `Y` | same-shape image tensors |
| `mask` | optional boolean tensor matching the input image shape |
| `win_size` | must be odd |

The function accepts 4D or 5D image tensors after squeezing singleton dimensions.

### Output behavior

Returns:

- masked scalar tensor when `mask` is present
- mean SSIM tensor when `size_average=True`
- per-image mean otherwise

### Mask behavior

When `mask` is present:

1. the code requires `size_average=True`
2. the mask is trimmed by `win_size // 2` on each border
3. aggregation is restricted to the selected pixels

This detail matters for any metric that depends on region-aware SSIM.

## `ms_ssim(...)`

```python
ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K=(0.01, 0.03),
) -> Tensor
```

This function performs multi-scale SSIM with repeated downsampling. It requires the image to be large enough for four downsampling stages.

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## `SSIM` Class

### Constructor

```python
SSIM(
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    channel: int = 3,
    spatial_dims: int = 2,
    K=(0.01, 0.03),
    nonnegative_ssim: bool = False,
)
```

### Public Methods

| Method | Purpose |
| --- | --- |
| `forward(X, Y, mask)` | Module wrapper around `ssim(...)`. |

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## `MS_SSIM` Class

### Constructor

```python
MS_SSIM(
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    channel: int = 3,
    spatial_dims: int = 2,
    weights: Optional[List[float]] = None,
    K=(0.01, 0.03),
)
```

### Public Methods

| Method | Purpose |
| --- | --- |
| `forward(X, Y)` | Module wrapper around `ms_ssim(...)`. |

## Practical Integration Example

This file is not usually configured directly. Instead, it is used through `SSIMMixin`, for example:

```yaml
metric_configs:
  ssim:
    pipe_name: ssim-pipe
    runtime_params:
      win_size: 11
      win_sigma: 1.5
```

## Failure Semantics

The low-level functions raise explicit errors for:

- mismatched input shapes
- mismatched mask shape
- unsupported tensor dimensionality
- even window sizes

`ms_ssim(...)` also asserts that the image is large enough for the configured number of downsampling stages.

## Extension Notes

- Change this file only when framework-wide SSIM behavior should change.
- Treat it as math infrastructure, not as a user-facing extension surface.
- If your change is only about one pipeline metric, the module layer is usually the better place to edit.

</div>
