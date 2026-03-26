# Face Analyzer Primitives

`face_analyzer.py` contains the reusable face-analysis helpers used by human-centric evaluation. It is implemented in `src/autopipeline/components/primitives/face_analyzer.py`.

This file mixes one registry-backed expert with two direct-analysis helpers.

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Primitive Group</span></div>

## Included Classes

| Class | Registry-backed | Main role |
| --- | --- | --- |
| `FaceDetectionMixin` | Yes, as `face-detector` | face-box discovery |
| `FaceMeshMixin` | No | face landmarks and alignment helpers |
| `FaceIDMixin` | No | face detection and embedding access through InsightFace |

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--expert">Expert</span></div>

## FaceDetectionMixin

### Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `face-detector` |
| Class | `FaceDetectionMixin` |

### Constructor

```python
FaceDetectionMixin(**kwargs)
```

#### Supported init kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `min_detection_confidence` | `0.5` | Confidence threshold for both MediaPipe detectors. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `detect_faces_adaptive(image_rgb)` | Try the long-range detector first, then the short-range fallback. |
| `get_first_face_bounding_box(image)` | Return the first detected face as a clamped pixel bbox. |

### Input / Output Contract

#### `detect_faces_adaptive(image_rgb)`

- input: RGB numpy array
- output: MediaPipe detection result object

#### `get_first_face_bounding_box(image)`

- input: PIL image
- output:
  - `(x1, y1, x2, y2)` tuple
  - `None` if no face is found

### Failure Semantics

This mixin uses soft failure:

- no face detected -> returns `None`

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

## FaceMeshMixin

### Constructor

```python
FaceMeshMixin(**kwargs)
```

#### Supported init kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `min_detection_confidence` | `0.5` | Detection confidence for MediaPipe Face Mesh. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `get_face_landmarks_from_cropped_face(face_crop)` | Extract landmark coordinates from a cropped face image. |
| `normalize_landmarks(L)` | Remove translation and scale. |
| `procrustes_align(L_ref, L)` | Align one normalized landmark set to another. |

### Input / Output Contract

#### `get_face_landmarks_from_cropped_face(face_crop)`

- input: cropped face as numpy array
- output:
  - `np.ndarray` of face landmarks in cropped-face coordinates
  - `None` if the crop is too small or mesh detection fails

#### `normalize_landmarks(L)`

- input: `(N, 2)` numpy array
- output: normalized `(N, 2)` array

#### `procrustes_align(L_ref, L)`

- input: two normalized landmark arrays
- output: aligned landmarks

### Failure Semantics

Soft-failure cases:

- crop height or width smaller than `10` -> returns `None`
- no face mesh found -> returns `None`

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

## FaceIDMixin

### Constructor

```python
FaceIDMixin(**kwargs)
```

#### Supported init kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `device` | `cpu` | Chooses CPU or CUDA execution provider. |
| `face_ana_name` | `buffalo_l` | InsightFace model pack name. |
| `model_root` | `~/.insightface` | Local model root. |
| `detection_size` | `(640, 640)` | Detection resolution passed to `FaceAnalysis.prepare(...)`. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `is_subject_face(face_bbox, sub_bbox)` | Decide whether a detected face belongs to the subject box. |
| `compute_two_faces_iou(boxA, boxB)` | IoU between two face boxes. |
| `detect_faces(image)` | Return all InsightFace detections for an image. |
| `get_highest_confidence_face(image)` | Return the face with the highest detection confidence. |

### Input / Output Contract

#### `detect_faces(image)`

- input: PIL image
- output: list of InsightFace face objects

#### `get_highest_confidence_face(image)`

- input: PIL image
- output:
  - one InsightFace face object
  - `None` if no faces are detected

### Failure Semantics

Soft-failure cases:

- no faces detected -> returns `None`

Initialization can fail if the required InsightFace assets are unavailable.

## Minimal Config Example

### Registry-backed face detector

```yaml
expert_configs:
  face-detector:
    min_detection_confidence: 0.5
```

### Face identity backend

```yaml
init_config:
  device: cpu
  face_ana_name: buffalo_l
  model_root: ${user_config.model_paths.arcface_root}
  detection_size: !tuple [640, 640]
```

## Extension Notes

- Add reusable face artifact logic here, not inside each metric module.
- Keep metric-specific score formulas in the module layer.
- Be explicit about whether a helper expects a full image or a cropped face region. That distinction matters throughout the human-centric stack.

</div>
