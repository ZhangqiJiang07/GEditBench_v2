# BaseClient

`BaseClient` is the abstract transport contract for prompt-driven modules in AutoPipeline. It is implemented in `src/autopipeline/components/primitives/clients/base_client.py`.

All concrete entries in `CLIENT_REGISTRY` inherit from this class.

<div class="doc-kind-row"><span class="doc-kind doc-kind--base">Base</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Role in the Stack

The client layer is intentionally narrow:

- clients handle transport
- prompt adapters format payloads
- modules interpret responses

`BaseClient` exists to keep that separation explicit.

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Class Signature

```python
class BaseClient(ABC):
    def __init__(self, model_name: str, **kwargs)
```

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor Parameters

| Parameter | Required | Meaning |
| --- | --- | --- |
| `model_name` | Yes | Model identifier passed to the backend transport layer. |
| `**kwargs` | No | Backend-specific config captured into `self.config`. |

### Stored attributes

The constructor stores:

- `self.model_name`
- `self.config`

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `call_model(messages)` | Abstract transport call implemented by subclasses. |
| `parse_response_to_json(response)` | Shared JSON extraction and repair helper. |

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Method Reference

### `call_model(messages)`

```python
call_model(messages) -> str
```

This method is abstract. Concrete subclasses define:

- accepted `messages` payload shape
- network or SDK call behavior
- retry logic
- returned raw text format

### `parse_response_to_json(response)`

```python
parse_response_to_json(response: str) -> Any
```

#### Input

| Parameter | Meaning |
| --- | --- |
| `response` | Raw text returned by the model backend. |

#### Output

Returns:

- a parsed Python object on success
- `None` if extraction and repair both fail

#### Internal behavior

The method:

1. tries to extract a JSON block from the response
2. repairs malformed JSON with `json_repair`
3. falls back to repairing the full response when no explicit block is found

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Integration Example

`BaseClient` is not instantiated directly, but all concrete client configs eventually map into its constructor shape:

```yaml
client_cfg:
  backend: api
  model_name: gpt-4o
  base_url: https://...
  api_key: ${client_config.api_key}
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

`BaseClient` itself does not perform transport retries, but `parse_response_to_json(...)` is soft-fail by design:

- malformed or missing JSON -> returns `None`
- parse errors are logged rather than raised

This behavior is important because the judge and parser modules often prefer retrying over crashing.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Subclass `BaseClient` when transport or authentication changes.
- Do not move prompt serialization into this layer.
- Keep output normalization above the client layer so transport code stays backend-specific and simple.

</div>

</div>
