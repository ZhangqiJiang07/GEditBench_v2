# Face Pipes

The face-related modules are implemented in `src/autopipeline/components/modules/face_pipe.py`.

They are documented together because they all consume face-local inputs prepared by the human-centric pipeline, but each class solves a distinct problem:

- geometry consistency
- texture consistency
- identity consistency

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Module Group</span></div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--helper">Helper</span></div>

## Shared Helper

### `face_crop_with_pad(...)`

```python
face_crop_with_pad(
    image: Image.Image,
    face_bbox: Tuple[int, int, int, int],
    pad_ratio: float = 0.1,
) -> np.ndarray
```

This helper:

1. converts the image to RGB numpy
2. expands the face bbox by `pad_ratio`
3. clamps the crop to image boundaries
4. returns the padded face crop

All three face pipes rely on this helper directly or conceptually.

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## FaceGeometryPipe

### Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `face-geometry-pipe` |
| Class | `FaceGeometryPipe` |
| Main mixins | `FaceMeshMixin` |
| Return type | `float` |

### Constructor

```python
FaceGeometryPipe(**kwargs)
```

#### Supported init kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `min_detection_confidence` | `0.5` | Forwarded to `FaceMeshMixin`. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `calc_L2_distance(landmarks1, landmarks2)` | Mean L2 distance across aligned landmark points. |
| `__call__(...)` | Crop faces, extract landmarks, align them, and measure drift. |

### Call Signature

```python
FaceGeometryPipe.__call__(
    cropped_ref_human_image: Image.Image,
    cropped_edited_human_image: Image.Image,
    ref_face_bbox: Tuple[int, int, int, int],
    edited_face_bbox: Tuple[int, int, int, int],
    metric: str = "L2_distance",
    **kwargs,
)
```

### Runtime Inputs

| Argument | Required | Meaning |
| --- | --- | --- |
| `cropped_ref_human_image` | Yes | Human crop from the reference image. |
| `cropped_edited_human_image` | Yes | Human crop from the edited image. |
| `ref_face_bbox` | Yes | Face box relative to the reference human crop. |
| `edited_face_bbox` | Yes | Face box relative to the edited human crop. |
| `metric` | Yes | Currently only `L2_distance`. |

#### Extra runtime kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `pad_ratio` | `0.1` | Extra padding applied around face crops. |

### Return Value

- `L2_distance` returns a float
- lower is better

If either side fails to produce landmarks, the current implementation returns `0.0`.

### Failure Semantics

- missing landmarks -> returns `0.0`
- unsupported metric -> `NotImplementedError`

### Minimal Integration Example

```yaml
metric_configs:
  face_geometry:
    pipe_name: face-geometry-pipe
    default_config: ${pipes_default.face-geometry-pipe}
    init_config:
      min_detection_confidence: 0.5
    scope: edit_area
```

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## FaceTexturePipe

### Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `face-texture-pipe` |
| Class | `FaceTexturePipe` |
| Main mixins | none beyond `BasePipe` |
| Return type | `float` |

### Constructor

```python
FaceTexturePipe(**kwargs)
```

This class defines no persistent init-time parameters.

### Public Methods

| Method | Purpose |
| --- | --- |
| `texture_resize(face_crop, texture_size)` | Resize a face crop to the texture-analysis resolution. |
| `high_freq(face_crop, texture_sigma)` | Extract a high-frequency residual by subtracting Gaussian blur. |
| `ab_hist(face_crop, texture_bins)` | Build LAB `a/b` histograms. |
| `texture_energy(face_crop)` | Compute Laplacian-based texture energy. |
| `__call__(...)` | Dispatch to one of the face-texture metrics. |

### Call Signature

```python
FaceTexturePipe.__call__(
    cropped_ref_human_image: Image.Image,
    cropped_edited_human_image: Image.Image,
    ref_face_bbox: Tuple[int, int, int, int],
    edited_face_bbox: Tuple[int, int, int, int],
    metric: str = "high_frequency_diff",
    **kwargs,
)
```

### Supported Metrics

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `high_frequency_diff` | mean absolute difference between high-frequency residuals | lower is better |
| `color_similarity` | histogram intersection in LAB color space | higher is better |
| `energy_ratio` | ratio of Laplacian texture energies | higher is better |

### Extra runtime kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `pad_ratio` | `0.1` | Extra padding around the face crop. |
| `texture_size` | `224` | Resize target before texture comparison. |
| `texture_sigma` | `3.0` | Gaussian blur sigma for high-frequency extraction. |
| `texture_bins` | `32` | Histogram bins for `color_similarity`. |

### Return Value

Returns a single float whose interpretation depends on the chosen metric branch.

### Failure Semantics

- unsupported metric -> `NotImplementedError`
- unlike the geometry pipe, this class does not add custom guards for invalid or empty crops, so valid face boxes are an important precondition

### Minimal Integration Example

```yaml
metric_configs:
  face_texture:
    pipe_name: face-texture-pipe
    init_config:
    scope: edit_area
    runtime_params:
      pad_ratio: 0.1
      texture_size: 224
      texture_sigma: 3.0
      texture_bins: 32
```

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## FaceIdentityPipe

### Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `face-identity-pipe` |
| Class | `FaceIdentityPipe` |
| Main mixins | `FaceIDMixin` |
| Return type | `float` or `None` |

### Constructor

```python
FaceIdentityPipe(**kwargs)
```

#### Supported init kwargs

| Key | Default | Meaning |
| --- | --- | --- |
| `device` | `cpu` | Selects CPU or CUDA provider for InsightFace. |
| `face_ana_name` | `buffalo_l` | InsightFace model pack name. |
| `model_root` | `~/.insightface` | Model storage root. |
| `detection_size` | `(640, 640)` | Detection resolution for `FaceAnalysis.prepare(...)`. |

### Public Methods

| Method | Purpose |
| --- | --- |
| `_compute_cosine_sim(feat1, feat2)` | Cosine similarity for face embeddings. |
| `_get_best_match_face_embedding_by_bbox(subject_crop, subject_face_bbox)` | Pick the detected face with the best IoU against a target bbox. |
| `compute_subject_face_consistency(...)` | Compare subject-face embeddings across cropped human regions. |
| `_union_coords_to_subject_bbox(coords)` | Merge multiple subject boxes into one enclosing bbox. |
| `compute_bg_face_consistency(ref_image, edited_image, coords)` | Compare only background faces, excluding the edited subject region. |
| `compute_max_match_face_consistency(ref_image, edited_image, coords)` | For each reference crop, search the best identity match in the edited image. |
| `__call__(...)` | Dispatch between identity modes. |

### Call Signature

```python
FaceIdentityPipe.__call__(
    ref_image: Image.Image = None,
    edited_image: Image.Image = None,
    coords: List[Tuple[int, int, int, int]] = None,
    cropped_ref_human_image: Image.Image = None,
    cropped_edited_human_image: Image.Image = None,
    ref_face_bbox: Tuple[int, int, int, int] = None,
    edited_face_bbox: Tuple[int, int, int, int] = None,
    metric: str = "bg_faceID_sim",
    **kwargs,
)
```

### Supported Metrics

| Metric | Required inputs | Meaning |
| --- | --- | --- |
| `bg_faceID_sim` | `ref_image`, `edited_image`, `coords` | preserve non-subject background face identity |
| `face_ID_sim` | cropped human images plus face boxes | preserve the main subject face identity |
| `max_match_face_ID_sim` | `ref_image`, `edited_image`, `coords` | search best edited-image face match for each reference crop |

### Return Value

Returns a float similarity score or `None`.

### Failure Semantics

Important soft-failure paths:

- missing `coords` for background modes -> returns `None`
- missing face boxes for `face_ID_sim` -> returns `None`
- no subject face embedding -> returns `0.0`
- no background faces in the reference image -> returns `None`

One important implementation caveat:

- `max_match_face_ID_sim` assumes each reference bbox crop still contains a detectable face; otherwise downstream access to `ref_face.embedding` is unsafe

### Minimal Config Example

```yaml
metric_configs:
  face_ID_sim:
    pipe_name: face-identity-pipe
    default_config: ${pipes_default.face-identity-pipe}
    init_config:
      device: cpu
      face_ana_name: buffalo_l
      model_root: ${user_config.model_paths.arcface_root}
      detection_size: !tuple [640, 640]
    scope: edit_area
```

## Extension Notes

- Keep reusable face artifact logic in `face_analyzer.py`.
- Use these pipes when the input already contains cropped human regions and face-local artifacts.
- If you add a new face metric, document whether it expects full images or cropped human images. That distinction is the main source of misuse in this file.

</div>
