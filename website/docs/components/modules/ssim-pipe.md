# SSIMPipe

`SSIMPipe` is registered as `ssim-pipe` and implemented in `src/autopipeline/components/modules/ssim_pipe.py`.

It is the structural-similarity module for image-edit evaluation. The pipe supports both:

- direct RGB SSIM
- luminance-only SSIM after histogram matching

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `ssim-pipe` |
| Class | `SSIMPipe` |
| Main mixins | `SSIMMixin`, `MaskProcessor` |
| Return type | `float` |

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor

```python
SSIMPipe(**kwargs)
```

### Module-specific init kwargs

None.

`SSIMPipe` does not define custom module-level initialization parameters. It relies on:

- `SSIMMixin` for the computation wrapper
- `MaskProcessor` for region masking

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `calc_RGB_channel_ssim(ref_image, edited_image, mask=None, win_size=7, win_sigma=1.5)` | Compute RGB SSIM on full images or masked regions. |
| `_get_L_channel_numpy(image)` | Convert an RGB image to the LAB luminance channel. |
| `calc_L_channel_ssim(ref_image, edited_image, mask=None, win_size=7, win_sigma=1.5)` | Compute SSIM on histogram-matched luminance only. |
| `__call__(...)` | Build the mask and dispatch to the requested metric branch. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Call Signature

```python
SSIMPipe.__call__(
    ref_image: Image.Image,
    edited_image: Image.Image,
    coords: List[Tuple[int, int, int, int]] = None,
    mask_mode: str = None,
    metric: str = "ssim",
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
| `coords` | No | Pixel-space boxes used to define the measured region. |
| `mask_mode` | No | `inner` or `outer`, typically derived from pipeline `scope`. |
| `metric` | Yes | `ssim` or `L_channel_ssim`. |

### Extra runtime kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `win_size` | `7` | SSIM Gaussian window size. |
| `win_sigma` | `1.5` | SSIM Gaussian window sigma. |

## Supported Metrics

| Metric | What it measures | Better direction |
| --- | --- | --- |
| `ssim` | RGB structural similarity | higher is better |
| `L_channel_ssim` | luminance-only similarity after histogram matching | higher is better |

### `ssim`

This branch:

1. converts both RGB images into `float32` tensors with shape `(1, 3, H, W)`
2. optionally builds a boolean mask with `MaskProcessor`
3. calls `SSIMMixin.compute(...)`

### `L_channel_ssim`

This branch:

1. converts both images from RGB to LAB
2. extracts the `L` channel
3. histogram-matches the edited luminance to the reference luminance
4. converts the result into single-channel tensors `(1, 1, H, W)`
5. reduces a 3-channel mask to one channel if necessary
6. calls `SSIMMixin.compute(...)`

This branch is especially useful when raw color or exposure shifts should not dominate the score.

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Return Value

The pipe returns a single float score.

- `1.0` is a near-perfect match
- lower values indicate stronger structural drift
- `SSIMMixin` may return `-1e8` if the low-level SSIM computation becomes `NaN`

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Examples

### RGB SSIM

```yaml
metric_configs:
  ssim:
    pipe_name: ssim-pipe
    init_config:
    scope: unedit_area
    runtime_params:
      win_size: 11
      win_sigma: 1.5
```

### Luminance-only SSIM

```yaml
metric_configs:
  L_channel_ssim:
    pipe_name: ssim-pipe
    init_config:
    scope: edit_area
    runtime_params:
      win_size: 11
      win_sigma: 1.5
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

The module itself has only one explicit hard-failure branch:

- unsupported metric -> `ValueError`

Most practical failure behavior comes from dependencies:

- invalid masks or bad image conversion fail upstream
- low-level `SSIMMixin.compute(...)` converts `NaN` into `-1e8`

This means the pipe is relatively stable, but the score interpretation depends on upstream preprocessing being correct.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Add a new branch here if you want another SSIM-style metric with the same general input contract.
- Keep mask generation aligned with `MaskProcessor` so pipeline `scope` continues to work consistently.
- If you change window defaults or luminance preprocessing, document the new metric interpretation because thresholding can shift significantly.

</div>

</div>
