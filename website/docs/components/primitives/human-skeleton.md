# HumanSkeletonMixin

`HumanSkeletonMixin` is the pose-estimation primitive implemented in `src/autopipeline/components/primitives/human_skeleton.py`.

It is a narrow reusable helper for producing pose landmarks used by body-structure metrics.

<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Class Signature

```python
HumanSkeletonMixin(**kwargs)
```

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor Parameters

| Key | Default | Meaning |
| --- | --- | --- |
| `static_mode` | `True` | Run MediaPipe Pose in static-image mode. |
| `model_complexity` | `0` | MediaPipe Pose model complexity. |
| `min_detection_confidence` | `0.5` | Detection confidence threshold. |

The implementation always sets:

- `enable_segmentation=False`

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `get_skeleton_np(image)` | Run pose estimation and return landmark coordinates. |

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Method Reference

```python
get_skeleton_np(image: Image.Image)
```

### Input

| Parameter | Meaning |
| --- | --- |
| `image` | PIL image containing a person. |

### Output

Returns:

- `np.ndarray` of pose landmarks with rows `[x, y, z, visibility]`
- `None` if no pose landmarks are found

The coordinates are normalized MediaPipe pose coordinates, not pixel-space coordinates.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Integration Example

```yaml
init_config:
  static_mode: true
  model_complexity: 1
  min_detection_confidence: 0.5
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

Soft-failure behavior:

- no pose landmarks -> returns `None`

Initialization or MediaPipe runtime failures are not wrapped in a custom recovery layer.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Keep general-purpose pose extraction here.
- Keep metric-specific body alignment or scoring above this layer.
- If you add a new body metric, document whether it expects normalized landmarks or pixel-space landmarks after post-processing.

</div>

</div>
