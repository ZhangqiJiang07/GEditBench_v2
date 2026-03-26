# GroundingDINOMixin

`GroundingDINOMixin` is the local zero-shot detection helper implemented in `src/autopipeline/components/primitives/grounding_dino.py`.

It is not registry-backed, but it is still a useful reusable primitive when a module needs direct text-conditioned object detection.

<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Class Signature

```python
GroundingDINOMixin(**kwargs)
```

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor Parameters

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `model_path` | No | `GroundingDINO/GroundingDINO-SwinB` | Hugging Face Grounding DINO checkpoint. |
| `device` | No | auto | Torch device used for inference. |

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `object_detection(image, text_labels, box_threshold=0.25, text_threshold=0.25)` | Run zero-shot detection and return the first processed result dict. |
| `get_bounding_boxes(image, text_labels, box_threshold=0.25, text_threshold=0.25)` | Convert raw detection results into a list of simple bbox dictionaries. |

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Method Reference

### `object_detection(...)`

```python
object_detection(
    image: Image.Image,
    text_labels: List[str],
    box_threshold=0.25,
    text_threshold=0.25,
)
```

#### Input

| Parameter | Meaning |
| --- | --- |
| `image` | PIL image to search. |
| `text_labels` | Object labels used as text prompts. |
| `box_threshold` | Detection threshold. |
| `text_threshold` | Text alignment threshold. |

#### Output

Returns the first processed result dict from the Hugging Face processor.

### `get_bounding_boxes(...)`

Returns a list of dictionaries with:

- `box`
- `score`
- `label`

Coordinates are rounded to integer pixel values.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Integration Example

This primitive is typically consumed from code rather than directly from YAML, but the equivalent init shape is:

```yaml
init_config:
  model_path: GroundingDINO/GroundingDINO-SwinB
  device: cuda
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

The file does not add explicit local exception handling.

That means:

- model-loading failures bubble up
- inference failures bubble up
- no detections usually result in empty structures rather than a custom sentinel value

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Use this mixin when you need local text-conditioned detection without the full parser-grounder runtime.
- Keep task-specific grounding orchestration in `ParserGrounderPipe`, not here.

</div>

</div>
