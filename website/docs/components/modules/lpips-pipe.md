# LPIPSPipe

`LPIPSPipe` is registered as `lpips-pipe` and implemented in `src/autopipeline/components/modules/lpips_pipe.py`.

It is the framework's perceptual-distance module for full-image or region-limited comparisons.

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `lpips-pipe` |
| Class | `LPIPSPipe` |
| Main mixins | `LPIPSMixin`, `MaskProcessor` |
| Return type | `float` |

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor

```python
LPIPSPipe(**kwargs)
```

### Supported init kwargs

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `device` | No | auto | Torch device for LPIPS inference. |
| `net` | No | `alex` | LPIPS backbone name passed to `lpips.LPIPS`. |

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `calc_lpips(ref_image, edited_image, mask=None)` | Compute LPIPS, optionally restricting the effective comparison region. |
| `__call__(...)` | Build the mask and dispatch to `calc_lpips`. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Call Signature

```python
LPIPSPipe.__call__(
    ref_image: Image.Image,
    edited_image: Image.Image,
    coords: List[Tuple[int, int, int, int]] = None,
    mask_mode: str = None,
    metric: str = "lpips",
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
| `coords` | No | Pixel-space boxes to measure or exclude. |
| `mask_mode` | No | `inner` or `outer`, normally derived from pipeline `scope`. |
| `metric` | Yes | Currently only `lpips`. |

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Mask Behavior

`LPIPSPipe` does not crop the image before inference. Instead, when a mask is present it:

1. fills the unmeasured region of the edited image with pixels from the reference image
2. computes LPIPS on the composited edited image against the original reference image
3. scales the resulting score by the measured-area fraction

This area normalization is implemented as:

```python
lpips_score / (computed_area / img_area)
```

when the mask contains at least one selected pixel.

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Return Value

The pipe returns a single `float`:

- lower is better
- `0.0` means perceptually very similar
- larger values indicate stronger perceptual drift

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Example

```yaml
metric_configs:
  lpips:
    pipe_name: lpips-pipe
    default_config: ${pipes_default.lpips-pipe}
    init_config:
      net: alex
    scope: unedit_area
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Modes

This module is relatively stable because it has few branching paths.

The main preconditions are:

- `coords` must be valid if `mask_mode` is used
- input images must already be valid RGB-like arrays when converted through `numpy`

Unsupported metrics raise:

```python
ValueError(f"Unsupported metric: {metric}")
```

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Reuse this pipe if you still want a perceptual-distance metric with region support.
- Add a new pipe only if the metric semantics are no longer LPIPS-like.
- If you change area normalization, document the new score interpretation because downstream thresholding may depend on it.

</div>

</div>
