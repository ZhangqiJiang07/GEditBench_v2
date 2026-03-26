# vLLMOnlineClient

`vLLMOnlineClient` is the local or internal OpenAI-compatible vLLM client in AutoPipeline. It is implemented in `src/autopipeline/components/primitives/clients/vllm_client.py` and registered as `vllm`.

<div class="doc-kind-row"><span class="doc-kind doc-kind--client">Client</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `vllm` |
| Class | `vLLMOnlineClient` |
| Base class | `BaseClient` |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Class Signature

```python
vLLMOnlineClient(
    model_name,
    **kwargs,
)
```

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor Parameters

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `model_name` | Yes | none | Model name served by the vLLM endpoint. |
| `ip_address` | Yes | none | Host where the vLLM server is running. |
| `port` | Yes | none | Port where the vLLM server exposes `/v1`. |
| `timeout` | No | `600` | Request timeout. |
| `max_tokens` | No | `2048` | Maximum generated tokens. |
| `retries` | No | `3` | Retry budget. |
| `temperature` | No | `0.7` | Sampling temperature. |
| `extra_body` | No | `{}` | Extra request fields forwarded for specific models. |

The constructor asserts that `ip_address` and `port` exist and then builds an `OpenAI` client with:

```text
http://<ip_address>:<port>/v1
```

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `call_model(messages)` | Execute an OpenAI-style chat completion against a vLLM server. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Callable Interface

```python
call_model(messages)
```

### Input contract

`messages` should already be OpenAI-style chat content, usually built by `OpenAIStylePromptAdapter`.

### Special behavior

If `"Qwen3-8B"` appears in `model_name`, the client forwards `extra_body` to the request. Each retry also changes the seed to:

```python
42 + try_count
```

### Return value

On success, the client returns:

```python
response.choices[0].message.content
```

On terminal failure, it returns `None`.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Example

```yaml
init_config:
  backend: vllm
  model_name: Qwen3-VL-8B-Instruct
  ip_address: ${client_config.ip_address}
  port: ${client_config.vlm_port}
  max_tokens: 2048
  retries: 3
  timeout: 600
  temperature: 0.7
```

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure Semantics

The retry loop:

- catches `requests.exceptions.RequestException`
- waits two seconds between attempts
- returns `None` after the retry budget is exhausted

Compared with `OpenAIAPIClient`, the exception handling is narrower and more transport-specific.

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Use this client when your local service is OpenAI-compatible.
- Do not overload it with non-compatible protocols.
- If a new server type is not wire-compatible with OpenAI, add a new client instead of branching this one indefinitely.

</div>

</div>
