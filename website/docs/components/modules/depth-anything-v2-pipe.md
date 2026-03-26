# DepthAnythingv2Pipe

`DepthAnythingv2Pipe` is registered as `depth-anything-v2-pipe` and implemented in `src/autopipeline/components/modules/depth_anything_v2_pipe.py`.

This module computes structural similarity in depth space rather than RGB space. It is useful when geometric consistency is more important than appearance similarity.

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `depth-anything-v2-pipe` |
| Class | `DepthAnythingv2Pipe` |
| Main mixins | `DepthAnythingv2Mixin`, `SSIMMixin`, `MaskProcessor` |
| Return type | `float` |

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor

```python
DepthAnythingv2Pipe(**kwargs)
```

### Supported init kwargs

| Key | Required | Meaning |
| --- | --- | --- |
| `model_path` | Yes in practice | Hugging Face depth-estimation checkpoint path. |
| `device` | No | Torch device for depth inference. |

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `_normalize_depth_map(depth_map)` | Rescale a raw depth map into `uint8` `[0, 255]`. |
| `calc_depth_ssim(...)` | Predict depth for both images and compute masked SSIM in depth space. |
| `__call__(...)` | Dispatch to the `depth_ssim` branch. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Call Signature

```python
DepthAnythingv2Pipe.__call__(
    ref_image: Image.Image,
    edited_image: Image.Image,
    coords: List[Tuple[int, int, int, int]] = None,
    mask_mode: str = None,
    metric: str = "depth_ssim",
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
| `coords` | No | Region boxes used for masking. |
| `mask_mode` | No | `inner` or `outer`, normally derived from pipeline `scope`. |
| `metric` | Yes | Currently only `depth_ssim`. |

### Extra runtime kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `resize_depth_maps` | `True` | Resize predicted depth back to original image size before scoring. |
| `win_size` | `7` | SSIM window size. |
| `win_sigma` | `1.5` | SSIM window sigma. |

## Supported Metric

| Metric | What it measures | Better direction |
| --- | --- | --- |
| `depth_ssim` | structural similarity between predicted depth maps | higher is better |

### Execution flow

`calc_depth_ssim(...)` performs the following steps:

1. predict depth maps for the reference and edited images
2. optionally resize them back to original image size
3. min-max normalize each depth map into `uint8`
4. build a mask with `MaskProcessor`
5. reduce the mask to one channel if necessary
6. compute SSIM on single-channel depth tensors

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Return Value

The pipe returns a single float score.

As with other SSIM-backed paths:

- higher is better
- `SSIMMixin.compute(...)` may degrade `NaN` to `-1e8`

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Example

```yaml
metric_configs:
  depth_ssim:
    pipe_name: depth-anything-v2-pipe
    default_config: ${pipes_default.depth-anything-v2-pipe}
    init_config:
    scope: edit_area
    runtime_params:
      win_size: 11
      win_sigma: 1.5
      resize_depth_maps: true
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

The module raises a `ValueError` for unsupported metrics.

Other important preconditions are implicit:

- `model_path` must be valid at initialization
- the predicted depth map should have non-degenerate range for stable normalization

The current `_normalize_depth_map(...)` implementation does not explicitly guard against a zero depth range, so extremely degenerate inputs rely on inherited SSIM fallback behavior.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Extend this pipe when the comparison remains "predict depth, then compare structure."
- If you want a different depth model but the same scoring logic, the change belongs mostly in `DepthAnythingv2Mixin`.
- If you want a different depth-space metric altogether, add a new metric branch and document the score direction clearly.

</div>

</div>
