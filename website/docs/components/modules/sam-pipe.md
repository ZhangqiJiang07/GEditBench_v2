# SAMPipe

`SAMPipe` is registered as `sam-pipe` and implemented in `src/autopipeline/components/modules/sam_pipe.py`.

This module is the direct SAM-backed region-overlap scorer. It compares whether the same bbox prompt yields similar segmentation masks before and after editing.

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `sam-pipe` |
| Class | `SAMPipe` |
| Main mixins | `SAMSegmentationMixin`, `MaskProcessor` |
| Return type | `float` or `None` |

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor

```python
SAMPipe(**kwargs)
```

### Supported init kwargs

| Key | Required | Meaning |
| --- | --- | --- |
| `model_cfg` | Yes in practice | SAM2 config path passed to `build_sam2(...)`. |
| `model_path` | Yes in practice | SAM2 checkpoint path. |
| `device` | No | Torch device for SAM inference. |

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `_compute_iou_in_single_bbox(mask1, mask2)` | Compute IoU between two binary masks. |
| `calc_iou(ref_image, edited_image, coords=None)` | Run SAM in each bbox and average IoU across valid regions. |
| `__call__(...)` | Dispatch to the `iou` metric branch. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Call Signature

```python
SAMPipe.__call__(
    ref_image: Image.Image,
    edited_image: Image.Image,
    coords: List[Tuple[int, int, int, int]] = None,
    mask_mode: str = None,
    metric: str = "iou",
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
| `coords` | Yes for useful output | Pixel-space boxes used as SAM prompts. |
| `mask_mode` | Ignored | Present only for signature consistency with other pipes. |
| `metric` | Yes | Currently only `iou`. |

## Supported Metric

| Metric | What it measures | Better direction |
| --- | --- | --- |
| `iou` | overlap between SAM masks extracted from the same bbox in both images | higher is better |

### Internal execution flow

For each bbox in `coords`, the pipe:

1. converts both images to RGB numpy arrays
2. runs `get_best_mask_in_bbox(...)` on the reference image
3. runs `get_best_mask_in_bbox(...)` on the edited image
4. computes IoU for that bbox
5. averages all valid per-bbox IoU scores

If the union of two masks is zero, `_compute_iou_in_single_bbox(...)` returns `1.0`.

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Return Value

The pipe returns:

- a float mean IoU if at least one bbox yields valid masks
- `None` if no valid IoU score can be computed

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Example

```yaml
metric_configs:
  region_iou:
    pipe_name: sam-pipe
    init_config:
      model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml
      model_path: /path/to/sam2.1_hiera_large.pt
      device: cuda
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

The module uses soft failure rather than exceptions for missing region artifacts:

- `coords is None` -> returns `None`
- one bbox fails to produce masks -> that bbox is skipped
- all bboxes fail -> returns `None`

Unsupported metrics raise:

```python
ValueError(f"Unsupported metric: {metric}")
```

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Keep this pipe focused on bbox-prompted segmentation overlap.
- If you need a new segmentation-based comparison but still rely on SAM masks, add a new metric branch here.
- If the segmentation backend changes entirely, that change belongs first in `SAMSegmentationMixin`.

</div>

</div>
