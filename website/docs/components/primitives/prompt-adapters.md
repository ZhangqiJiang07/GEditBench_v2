# Prompt Adapters

`prompt_adapters.py` contains the payload builders that translate a `PromptTemplate` into the backend-specific message format expected by a client. It is implemented in `src/autopipeline/components/primitives/prompt_adapters.py`.

These classes are registered in `PROMPT_ADAPTER_REGISTRY`.

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Primitive Group</span></div>

## Registry Entries

| Registry key | Class | Typical backend |
| --- | --- | --- |
| `openai_style` | `OpenAIStylePromptAdapter` | OpenAI-compatible APIs and vLLM |
| `google_style` | `GoogleGenAIStylePromptAdapter` | Google GenAI / Gemini backends |

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--shared">Shared Infra</span></div>

## Shared Symbols

| Symbol | Purpose |
| --- | --- |
| `SEPARATOR_RULES` | Defines separators between adjacent block types. |
| `get_separator(prev_type, current_type)` | Resolve the separator string for two neighboring content blocks. |
| `BasePromptAdapter` | Abstract payload-builder contract. |

## Block Contract

All adapters assume the prompt template exposes:

- `template.system_prompt`
- `template.render_blocks(**kwargs)`

The rendered blocks must contain only supported block types:

- `{"type": "text", "content": "..."}`
- `{"type": "image", "content": ImageWrapper}`

## Separator Rules

`get_separator(...)` uses these explicit transitions:

| Transition | Separator |
| --- | --- |
| `text -> image` | `"\n"` |
| `image -> text` | `"\n\n"` |
| `image -> image` | `"\n\n"` |
| everything else | `"\n"` |

This logic exists because multimodal models can be sensitive to how text and image blocks are separated.

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--base">Base</span></div>

## BasePromptAdapter

### Class Signature

```python
class BasePromptAdapter(ABC):
    def build_payload(self, template, **kwargs) -> list
```

### Public Methods

| Method | Purpose |
| --- | --- |
| `build_payload(template, **kwargs)` | Abstract method implemented by concrete adapters. |

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--adapter">Adapter</span></div>

## OpenAIStylePromptAdapter

### Registry key

`openai_style`

### Public Methods

| Method | Purpose |
| --- | --- |
| `build_payload(template, **kwargs)` | Build OpenAI-style chat messages. |

### Output format

The adapter returns a `messages` list containing:

- an optional system message
- one user message whose `content` is a list of text and image items

Image blocks are serialized as data URLs through:

```python
img_wrapper.as_data_url()
```

### Typical output shape

```python
[
  {"role": "system", "content": "..."},
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }
]
```

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--adapter">Adapter</span></div>

## GoogleGenAIStylePromptAdapter

### Registry key

`google_style`

### Public Methods

| Method | Purpose |
| --- | --- |
| `build_payload(template, **kwargs)` | Build Gemini-style `parts` content. |

### Output format

The adapter returns a `parts` list.

Behavior:

- the first text block can absorb the system prompt
- image blocks are turned into `types.Part.from_bytes(...)`
- separators are inserted between adjacent blocks

Image blocks are serialized through:

```python
img_wrapper.as_bytes()
```

## Practical Selection Rule

In the current runtime, prompt style is backend-driven:

- Google backends -> `google_style`
- all other current client-backed paths -> `openai_style`

This is why client and prompt-adapter selection must stay coordinated.

## Minimal Integration Example

Prompt adapters are not usually selected directly in YAML. The selection happens indirectly through the backend:

```yaml
init_config:
  backend: google
  model_name: gemini-3-pro-native
  prompt_info:
    prompt_id: vlm/assessment/visual_consistency/pairwise
    version: v1
```

The runtime will infer `google_style` from `backend: google`.

## Failure Semantics

This file does not add explicit validation around:

- unsupported block types
- malformed `ImageWrapper` content
- incompatible prompt templates

Those issues will generally surface as runtime exceptions in the adapter or downstream client.

## Extension Notes

- Add a new adapter when payload shape changes.
- Keep network transport inside clients.
- Keep prompt semantics inside prompt assets.
- Keep output interpretation inside modules.

</div>
