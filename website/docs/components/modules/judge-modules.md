# Judge Modules

The judge modules are the prompt-driven comparison and scoring units implemented in `src/autopipeline/components/modules/judge.py`.

They are documented together because both classes share the same client stack, the same prompt-adapter selection logic, and the same schema-validation path.

<div class="doc-kind-row"><span class="doc-kind doc-kind--group">Module Group</span></div>

## Included Registry Entries

| Registry key | Class | Primary output |
| --- | --- | --- |
| `pairwise-judge` | `PairwiseJudge` | normalized winner: `Image A`, `Image B`, `Tie`, or `Failed` |
| `viescore` | `VIEscorePipe` | single numeric score or pairwise winner derived from scores |

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--shared">Shared Infra</span></div>

## Shared Infrastructure

### `build_client(...)`

`build_client(...)` resolves a backend from `CLIENT_REGISTRY` and forwards backend-specific fields to the client constructor.

#### Recognized client config keys

| Key | Meaning |
| --- | --- |
| `backend` | registry key such as `api`, `google`, or `vllm` |
| `model_name` | model identifier passed to the backend |
| `ip_address`, `port` | used by the vLLM client |
| `base_url`, `api_key` | used by HTTP- or SDK-based remote clients |
| extra kwargs | forwarded to the selected client |

### `ClientPipe`

Both judge modules inherit from `ClientPipe`.

```python
ClientPipe(
    prompt_template: PromptTemplate,
    client_cfg: Dict[str, Any],
)
```

#### What `ClientPipe` does

`ClientPipe` is responsible for:

1. building the backend client
2. selecting `google_style` or `openai_style` payload formatting
3. resolving optional input and output schemas from prompt metadata
4. validating or cleaning input before a model call

#### Shared constructor parameters

| Parameter | Required | Meaning |
| --- | --- | --- |
| `prompt_template` | Yes | Prompt asset loaded from `PromptAssetStore`. |
| `client_cfg` | Yes | Backend connection config and prompt metadata. |

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## PairwiseJudge

`PairwiseJudge` is the canonical A/B evaluator.

### Constructor

```python
PairwiseJudge(
    prompt_template: PromptTemplate,
    client_cfg: Dict[str, Any],
)
```

It adds no extra constructor arguments beyond `ClientPipe`.

### Public Methods

| Method | Purpose |
| --- | --- |
| `_is_valid_winner(winner_str)` | Normalize common winner aliases into `Image A`, `Image B`, or `Tie`. |
| `__call__(input_dict, **kwargs)` | Execute the comparison prompt and return a normalized result dict. |

### Call Signature

```python
PairwiseJudge.__call__(
    input_dict: Dict[str, Any],
    **kwargs,
)
```

### Expected input fields

If the prompt declares `PairJudgeInput`, the input schema is:

| Field | Type | Meaning |
| --- | --- | --- |
| `instruction` | `str` | Edit instruction being judged. |
| `input_image` | `Any` | Source image. |
| `edited_images` | `list[Any]` | Exactly two candidate edited images. |

### Output format

The return value is always a dict:

```python
{
  "type": "pairwise_comparison",
  "value": "Image A" | "Image B" | "Tie" | "Failed",
  "meta": {
    "raw_response": "<model text>"
  }
}
```

### Winner normalization rules

The following raw values are accepted:

- `image_a`, `image a`, `a` -> `Image A`
- `image_b`, `image b`, `b` -> `Image B`
- `tie`, `equal`, `both`, `none` -> `Tie`

Anything else is treated as invalid and retried.

### Failure behavior

If the model output cannot be parsed or does not validate after retries:

- `value` stays `"Failed"`
- the last raw response is still preserved in `meta.raw_response`

This is a soft failure, not an exception.

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

</div>

<div class="doc-section-card">
<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

## VIEscorePipe

`VIEscorePipe` uses the same client path but converts model output into numeric scores.

### Constructor

```python
VIEscorePipe(
    prompt_template: PromptTemplate,
    client_cfg: Dict[str, Any],
)
```

It also adds no extra constructor arguments beyond `ClientPipe`.

### Public Methods

| Method | Purpose |
| --- | --- |
| `_parse_viescore(json_response)` | Convert prompt output into a float score. |
| `score_single_input(messages)` | Retry until one image has a valid score or the retry budget is exhausted. |
| `__call__(input_dict, **kwargs)` | Score one or two edited images and return either a scalar or a pairwise winner. |

### Call Signature

```python
VIEscorePipe.__call__(
    input_dict: Dict[str, Any],
    **kwargs,
)
```

### Expected input fields

The runtime input must contain:

| Field | Type | Meaning |
| --- | --- | --- |
| `instruction` | `str` | Edit instruction. |
| `input_image` | `Any` | Source image. |
| `edited_images` | `list[Any]` | One or two edited images. |

The code asserts that the list length is either `1` or `2`.

### Output format

For one edited image:

```python
{
  "type": "single_score",
  "value": <float> | "Failed",
  "meta": {
    "vie_score": [<float | None>],
    "raw_response": {"edited_image_0": "<model text>"}
  }
}
```

For two edited images:

```python
{
  "type": "pairwise_comparison",
  "value": "Image A" | "Image B" | "Tie" | "Failed",
  "meta": {
    "vie_score": [score_a, score_b],
    "raw_response": {
      "edited_image_0": "<model text>",
      "edited_image_1": "<model text>"
    }
  }
}
```

### Score parsing behavior

`_parse_viescore(...)` supports two prompt families:

| Prompt behavior | Expected `score` field | Parsing rule |
| --- | --- | --- |
| UnicEdit-style prompts | scalar | cast directly to `float` |
| EditScore-style `v2` prompts | list | choose index based on prompt id family |

For `v2` prompts:

- `instruction_following` -> `score[0]`
- `visual_quality` -> `min(score)`
- otherwise -> `score[1]`

### Failure behavior

If any score cannot be parsed:

- that image keeps `None` in `meta.vie_score`
- the overall result remains `"Failed"`
- raw responses are still returned

## Minimal Config Example

```yaml
metric_configs:
  pair-judge:
    pipe_name: pairwise-judge
    default_config: ${pipes_default.pairwise-judge}
    init_config:
      backend: api
      model_name: gpt-4o
      api_key: ${client_config.api_key}
      base_url: ${client_config.base_url}
      prompt_info:
        prompt_id: vlm/assessment/visual_consistency/pairwise
        version: v1
```

For `viescore`, replace `pipe_name` with `viescore` and point `prompt_info` at a `.../viescore/...` prompt asset.

## Extension Notes

- Add a new client if transport changes.
- Add a new prompt adapter if payload shape changes.
- Reuse `PairwiseJudge` if the semantics are still "pick A, B, or Tie."
- Reuse `VIEscorePipe` if the semantics are still "produce per-image scalar scores."
- Only create a new judge module when the return type or retry logic genuinely differs from these two patterns.

</div>
