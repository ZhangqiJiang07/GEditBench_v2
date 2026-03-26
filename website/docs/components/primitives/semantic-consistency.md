# Semantic Consistency Mixins

`semantic_consistency.py` contains the reusable embedding backbones for semantic similarity. It is implemented in `src/autopipeline/components/primitives/semantic_consistency.py`.

These mixins provide the feature-extraction layer used by `CLIPPipe`, `DINOv3Pipe`, and `BodyAppearancePipe`.

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Primitive Group</span></div>

## Included Classes

| Class | Primary role | Typical consumers |
| --- | --- | --- |
| `CLIPMixin` | CLIP patch and CLS embeddings | `clip-pipe` |
| `DINOv3Mixin` | DINOv3 patch and CLS embeddings | `dino-v3-pipe`, `body-appearance-pipe` |

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

## CLIPMixin

### Class Signature

```python
CLIPMixin(**kwargs)
```

### Constructor Parameters

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `model_path` | No | `openai/clip-vit-base-patch32` | CLIP vision checkpoint. |
| `device` | No | auto | Torch device used for inference. |

### Derived attributes

The mixin derives:

- `img_input_size`
- `patch_size`

from `EMBED_MODEL_RESOLUTION`, with a fallback of `(224, 32)`.

### Public Methods

| Method | Purpose |
| --- | --- |
| `get_features(image, mask=None)` | Return patch embeddings, optionally filtered by a boolean mask. |
| `get_cls_feature(image)` | Return the CLS embedding. |

### Input / Output Contract

#### `get_features(image, mask=None)`

- input:
  - PIL image
  - optional boolean mask over patches
- output:
  - patch embeddings with the CLS token removed
  - if `mask` is provided, returns only `embeddings[mask, :]`

#### `get_cls_feature(image)`

- input: PIL image
- output: one CLS embedding tensor

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--mixin">Mixin</span></div>

## DINOv3Mixin

### Class Signature

```python
DINOv3Mixin(**kwargs)
```

### Constructor Parameters

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `model_path` | Yes in practice | `None` in code | DINOv3 checkpoint path. |
| `device` | No | auto | Torch device used for inference. |

### Important implementation caveat

Although the code uses `kwargs.get("model_path", None)`, it immediately calls:

```python
model_path.split("/")[-1]
```

So `model_path` is effectively required.

### Derived attributes

The mixin derives:

- `input_image_size`
- `patch_size`

from `EMBED_MODEL_RESOLUTION`, with a fallback of `(224, 16)`.

### Public Methods

| Method | Purpose |
| --- | --- |
| `get_features(image, mask=None)` | Return DINO patch embeddings after removing special tokens. |
| `get_cls_feature(image)` | Return the CLS embedding. |

### Input / Output Contract

#### `get_features(image, mask=None)`

- input:
  - PIL image
  - optional boolean mask over patches
- output:
  - patch embeddings with both:
    - the CLS token
    - DINO register tokens
    removed

#### `get_cls_feature(image)`

- input: PIL image
- output: one CLS embedding tensor

## Minimal Config Example

### CLIP-based module init

```yaml
init_config:
  model_path: openai/clip-vit-base-patch32
  device: cuda
```

### DINO-based module init

```yaml
init_config:
  model_path: ${user_config.model_paths.dino_v3_path}
  device: cuda
```

## Failure Semantics

Neither mixin adds a local recovery layer around:

- Hugging Face model loading
- processor construction
- tensor indexing with incompatible masks

These failures bubble up to the module layer.

## Extension Notes

- Use this file to change feature backbones or token-handling logic.
- Keep actual score formulas in the module layer.
- If you add a new embedding backbone, follow the same split:
  - patch-level feature method
  - CLS feature method
  - explicit input-size / patch-size metadata

</div>
