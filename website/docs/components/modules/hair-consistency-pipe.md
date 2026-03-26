# HairConsistencyPipe

`HairConsistencyPipe` is registered as `hair-consistency-pipe` and implemented in `src/autopipeline/components/modules/hair_pipe.py`.

This module compares appearance consistency only within hair regions. It assumes hair masks have already been prepared by the human-centric pipeline.

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `hair-consistency-pipe` |
| Class | `HairConsistencyPipe` |
| Main dependencies | hair masks prepared by `hair-segmenter` |
| Return type | `float` or `None` |

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor

```python
HairConsistencyPipe(**kwargs)
```

The class does not define any constructor-time configuration.

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `color_hist_distance(Io, Mo, Ie, Me, bins=32, eps=1e-6)` | Compare LAB histogram distance inside masked hair regions. |
| `texture_energy_diff(Io, Mo, Ie, Me, eps=1e-6)` | Compare Laplacian-based texture energy inside hair regions. |
| `high_freq_diff(Io, Mo, Ie, Me, eps=1e-6)` | Compare Sobel high-frequency magnitude inside hair regions. |
| `__call__(...)` | Dispatch to one of the hair metrics. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Call Signature

```python
HairConsistencyPipe.__call__(
    cropped_ref_human_image: Image.Image,
    cropped_edited_human_image: Image.Image,
    ref_hair_mask: np.ndarray = None,
    edited_hair_mask: np.ndarray = None,
    metric: str = "color_distance",
    **kwargs,
)
```

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Runtime Inputs

| Argument | Required | Meaning |
| --- | --- | --- |
| `cropped_ref_human_image` | Yes | Reference human crop. |
| `cropped_edited_human_image` | Yes | Edited human crop. |
| `ref_hair_mask` | Yes for useful output | Binary hair mask aligned to the reference crop. |
| `edited_hair_mask` | Yes for useful output | Binary hair mask aligned to the edited crop. |
| `metric` | Yes | `color_distance`, `texture_energy_diff`, or `high_frequency_diff`. |

### Extra runtime kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `texture_bins` | `32` | Histogram bins for `color_distance`. |

## Supported Metrics

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `color_distance` | chi-square LAB histogram distance | lower is better |
| `texture_energy_diff` | relative Laplacian energy difference | lower is better |
| `high_frequency_diff` | relative Sobel high-frequency difference | lower is better |

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Return Value

Returns a float distance or difference score.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Example

```yaml
metric_configs:
  hair_high_frequency_diff:
    pipe_name: hair-consistency-pipe
    default_config:
    scope: edit_area
```

If you want the color metric explicitly:

```yaml
metric_configs:
  hair_color_distance:
    pipe_name: hair-consistency-pipe
    init_config:
    scope: edit_area
    runtime_params:
      texture_bins: 32
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

The main soft-failure branch is:

- either hair mask is missing -> returns `None`

Unsupported metrics raise:

```python
ValueError(f"Unsupported metric: {metric}")
```

One practical caveat is that the code checks only for missing masks, not for empty masks. A non-null mask with zero positive area can still make downstream `numpy` reductions unstable. In practice, the upstream expert should ensure the masks are meaningful.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Keep reusable hair-region extraction in the expert layer.
- Add a new metric here if the comparison is still region-local and mask-driven.
- Document score direction carefully, because all current hair metrics are distance-style and interpreted as "lower is better."

</div>

</div>
