# OpenAIAPIClient

`OpenAIAPIClient` is the OpenAI-compatible HTTP client in AutoPipeline. It is implemented in `src/autopipeline/components/primitives/clients/openai_client.py` and registered in `CLIENT_REGISTRY` as `api`.

<div class="doc-kind-row"><span class="doc-kind doc-kind--client">Client</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `api` |
| Class | `OpenAIAPIClient` |
| Base class | `BaseClient` |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Class Signature

```python
OpenAIAPIClient(
    model_name: str = "gpt-4o",
    **kwargs,
)
```

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor Parameters

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `model_name` | No | `gpt-4o` | Model name sent in the OpenAI-style request body. |
| `base_url` | Yes | none | Full chat-completions endpoint URL. |
| `api_key` | Yes | none | Bearer token used in the `Authorization` header. |
| `max_tokens` | No | `2048` | Maximum generated tokens. |
| `retries` | No | `3` | Retry budget after request failures. |
| `timeout` | No | `600` | Request timeout in seconds. |

The constructor asserts that `base_url` and `api_key` are present.

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `call_model(messages)` | Execute one OpenAI-style chat-completions HTTP request with retries. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Callable Interface

```python
call_model(messages) -> str | None
```

### Input contract

`messages` must already be an OpenAI-style chat payload, typically produced by `OpenAIStylePromptAdapter`.

### Request body

The client sends:

```json
{
  "model": "<model_name>",
  "stream": false,
  "max_tokens": 2048,
  "messages": [...]
}
```

### Return value

On success, the method returns:

```python
resp.json()["choices"][0]["message"]["content"].strip()
```

On terminal failure, it returns `None`.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Example

```yaml
init_config:
  backend: api
  model_name: gpt-4o
  base_url: ${client_config.base_url}
  api_key: ${client_config.api_key}
  max_tokens: 2048
  retries: 3
  timeout: 600
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

The method retries on any exception:

- sleeps for two seconds between attempts
- increments an internal retry counter
- returns `None` after the retry budget is exhausted

Because the retry loop catches broad exceptions, caller code should treat `None` as the canonical transport failure signal.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Reuse this client when your backend already speaks the OpenAI chat-completions protocol.
- Replace it only if the wire format or response extraction logic changes materially.
- Keep prompt formatting in `prompt_adapters.py`, not here.

</div>

</div>
