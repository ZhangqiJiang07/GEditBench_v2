# GoogleAPIClient

`GoogleAPIClient` is the Google GenAI backend client in AutoPipeline. It is implemented in `src/autopipeline/components/primitives/clients/google_client.py` and registered in `CLIENT_REGISTRY` as `google`.

<div class="doc-kind-row"><span class="doc-kind doc-kind--client">Client</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `google` |
| Class | `GoogleAPIClient` |
| Base class | `BaseClient` |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Class Signature

```python
GoogleAPIClient(
    model_name="gemini-3-pro-native",
    **kwargs,
)
```

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor Parameters

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `model_name` | No | `gemini-3-pro-native` | Target model name for `generate_content`. |
| `base_url` | Yes | none | Base URL for the configured Google-compatible endpoint. |
| `api_key` | Yes | none | API key for the SDK client. |
| `api_version` | No | `v1alpha` | API version passed through HTTP options. |
| `max_tokens` | No | `2048` | Maximum output tokens. |
| `retries` | No | `3` | Retry budget. |

The constructor asserts that `base_url` and `api_key` exist, then builds a `genai.Client`.

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `call_model(messages)` | Execute Gemini-style content generation with retry logic. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Callable Interface

```python
call_model(messages)
```

### Input contract

`messages` should already be Gemini-style `contents`, typically produced by `GoogleGenAIStylePromptAdapter`.

### Return value

On success, the method returns `response.text`.

If no valid response text is produced after retries, it returns `None`.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Example

```yaml
init_config:
  backend: google
  model_name: gemini-3-pro-native
  base_url: ${client_config.base_url}
  api_key: ${client_config.api_key}
  api_version: v1alpha
  max_tokens: 2048
  retries: 3
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

The retry loop:

- retries `range(self.retries)`
- uses exponential backoff with jitter
- treats `429` and exhausted-resource cases as retriable
- logs empty responses and transport failures

Terminal failure returns `None`.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Use this client when the runtime is genuinely Google GenAI / Gemini-shaped.
- Keep `parts` formatting in the prompt adapter, not in this client.
- If SDK behavior changes, this file is the right place to isolate that change from the module layer.

</div>

</div>
