# Body Pipes

The body-related modules are implemented in `src/autopipeline/components/modules/human_body.py`.

They are documented together because both classes operate on body-local crops and body masks, but the scoring targets are different:

- pose and silhouette consistency
- appearance consistency

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Module Group</span></div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--helper">Helper</span></div>

## Shared Helpers

### `pad_face_bbox(...)`

```python
pad_face_bbox(
    face_bbox: Tuple[int, int, int, int],
    pad_ratio: float = 0.3,
    image_h: int = None,
    image_w: int = None,
) -> Tuple[int, int, int, int] | None
```

This helper expands a face bbox and clamps it to image boundaries. It is used to remove the face region from body masks before body-only comparison.

### `get_cleaned_resized_body_mask(...)`

```python
get_cleaned_resized_body_mask(
    body_mask,
    hair_mask=None,
    face_bbox=None,
    target_h=None,
    target_w=None,
) -> np.ndarray
```

This helper:

1. starts from the body mask
2. optionally removes hair pixels
3. optionally removes the padded face region
4. optionally resizes the result

It is central to both body-related modules.

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## BodyPoseAndShapePipe

### Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `body-pose-and-shape-pipe` |
| Class | `BodyPoseAndShapePipe` |
| Main mixins | `HumanSkeletonMixin` |
| Return type | `float` or `None` |

### Constructor

```python
BodyPoseAndShapePipe(**kwargs)
```

#### Supported init kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `static_mode` | `True` | MediaPipe Pose inference mode. |
| `model_complexity` | `0` | MediaPipe Pose model complexity. |
| `min_detection_confidence` | `0.5` | Pose detection confidence threshold. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `calc_skeleton_aligned_shape_iou(...)` | Align body masks through skeleton-derived affine normalization and compute IoU. |
| `calc_pose_position_error(...)` | Compute normalized keypoint-position error after Procrustes-style normalization. |
| `__call__(...)` | Dispatch between body-shape and body-pose metrics. |

### Call Signature

```python
BodyPoseAndShapePipe.__call__(
    cropped_ref_human_image: Image.Image,
    cropped_edited_human_image: Image.Image,
    ref_face_bbox: Tuple[int, int, int, int] = None,
    edited_face_bbox: Tuple[int, int, int, int] = None,
    ref_hair_mask: np.ndarray = None,
    edited_hair_mask: np.ndarray = None,
    ref_body_mask: np.ndarray = None,
    edited_body_mask: np.ndarray = None,
    metric: str = "body_shape_iou",
    **kwargs,
)
```

### Supported Metrics

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `body_shape_iou` | silhouette overlap after skeleton-aware alignment | higher is better |
| `body_pose_position_error` | normalized joint-position drift | lower is better |

### Extra runtime kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `face_bbox_pad_ratio` | `0.3` | Padding ratio when removing face from the body mask. |
| `target_canvas_size` | `(512, 512)` | Target canvas for aligned body-shape IoU. |

### Return Value

Returns a float score or `None`.

### Failure Semantics

Important soft-failure behavior:

- missing body mask -> returns `None`
- missing pose landmarks -> returns `None`
- insufficient valid keypoints:
  - shape metric falls back to direct mask IoU
  - pose metric returns `None`

Unsupported metrics raise `ValueError`.

### Minimal Config Example

```yaml
metric_configs:
  body_appearance_dino_cosine_sim:
    pipe_name: body-appearance-pipe
    default_config: ${pipes_default.body-appearance-pipe}
    init_config:
    scope: edit_area
    runtime_params:
      face_bbox_pad_ratio: 0.3
      patch_mask_threshold: 0.1
```

For this specific class, a shape-oriented config would look like:

```yaml
metric_configs:
  body_shape_iou:
    pipe_name: body-pose-and-shape-pipe
    default_config: ${pipes_default.body-pose-and-shape-pipe}
    init_config:
      static_mode: true
      model_complexity: 1
      min_detection_confidence: 0.5
    scope: edit_area
    runtime_params:
      face_bbox_pad_ratio: 0.3
      target_canvas_size: !tuple [512, 512]
```

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## BodyAppearancePipe

### Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `body-appearance-pipe` |
| Class | `BodyAppearancePipe` |
| Main mixins | `DINOv3Mixin`, `MaskProcessor` |
| Return type | `float` or `None` |

### Important inheritance note

`BodyAppearancePipe` is registered and usable, but unlike most module classes it does not explicitly inherit `BasePipe`. That is a real implementation detail, not just a documentation abstraction. If you extend it, read the class carefully instead of assuming it follows the exact same inheritance pattern as every other module.

### Constructor

```python
BodyAppearancePipe(**kwargs)
```

#### Supported init kwargs

| Key | Required | Meaning |
| --- | --- | --- |
| `model_path` | Yes in practice | DINOv3 checkpoint path. |
| `device` | No | Torch device for DINO inference. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `calc_dino_aggregation_cosine_similarity(...)` | Average selected DINO features per image and compare them by cosine similarity. |
| `__call__(...)` | Build cleaned body patch masks and compute appearance similarity. |

### Call Signature

```python
BodyAppearancePipe.__call__(
    cropped_ref_human_image: Image.Image,
    cropped_edited_human_image: Image.Image,
    ref_face_bbox: Tuple[int, int, int, int] = None,
    edited_face_bbox: Tuple[int, int, int, int] = None,
    ref_hair_mask: np.ndarray = None,
    edited_hair_mask: np.ndarray = None,
    ref_body_mask: np.ndarray = None,
    edited_body_mask: np.ndarray = None,
    metric: str = "body_appearance_dino_cosine_sim",
    **kwargs,
)
```

### Supported Metric

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `body_appearance_dino_cosine_sim` | similarity of aggregated DINO body-region features | higher is better |

### Extra runtime kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `face_bbox_pad_ratio` | `0.3` | Padding ratio when removing face from the body region. |
| `patch_mask_threshold` | `0.1` | Threshold when converting cleaned body masks into patch masks. |

### Return Value

Returns a float similarity score clipped with:

```python
max(0.0, score)
```

### Failure Semantics

The main soft-failure branches are:

- missing body masks -> returns `None`
- empty selected patch set on either side -> returns `None`

Unsupported metrics raise `ValueError`.

### Minimal Config Example

```yaml
metric_configs:
  body_appearance_dino_cosine_sim:
    pipe_name: body-appearance-pipe
    default_config: ${pipes_default.body-appearance-pipe}
    init_config:
    scope: edit_area
    runtime_params:
      face_bbox_pad_ratio: 0.3
      patch_mask_threshold: 0.1
```

## Extension Notes

- Keep shared body-mask cleanup in the helper layer instead of duplicating it inside each metric branch.
- Document whether a new metric is shape-oriented or appearance-oriented, because the expected score direction differs.
- If a new metric still depends on cleaned body masks and DINO features, adding a branch to `BodyAppearancePipe` is usually cleaner than creating a brand new module.

</div>
