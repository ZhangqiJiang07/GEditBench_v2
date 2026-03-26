# ParserGrounderPipe

`ParserGrounderPipe` is the region-discovery front end for AutoPipeline. It is registered as `parser-grounder` and implemented in `src/autopipeline/components/modules/parser_grounder.py`.

This is the module that turns a natural-language edit instruction into structured edit targets and pixel-space bounding boxes. If you need region-aware metrics such as `sam_clip_cls_sim`, `sam_dino_cls_sim`, or any mask-based object-centric measurement, this pipe is usually the first prerequisite.

<div class="doc-kind-row"><span class="doc-kind doc-kind--class">Class</span></div>

<div class="doc-section-card">

<div class="doc-api-block doc-api-block--overview">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--overview">Overview</span></div>

## Registry Entry

| Field | Value |
| --- | --- |
| Registry key | `parser-grounder` |
| Class | `ParserGrounderPipe` |
| Selected from | `parser_grounder_config` |
| Return type | `tuple[dict | None, list[tuple[int, int, int, int]] | None]` |

</div>

<div class="doc-api-block doc-api-block--constructor">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--constructor">Constructor</span></div>

## Constructor

```python
ParserGrounderPipe(
    config,
    prompt_template_dict: Dict[str, PromptTemplate],
)
```

### Parameters

| Parameter | Type | Required | Meaning |
| --- | --- | --- | --- |
| `config` | `dict` | Yes | Parser-grounder config block containing `instruction_parser` and `general_grounder`. |
| `prompt_template_dict` | `dict[str, PromptTemplate]` | Yes | Prompt assets keyed by `instruction_parser` and `general_grounder`. |

### Required config shape

The constructor expects both sub-blocks to exist:

- `config["instruction_parser"]["init_config"]`
- `config["general_grounder"]["init_config"]`

Each `init_config` is passed to `build_client(...)`, so it usually contains:

- `backend`
- `model_name`
- backend-specific connection fields such as `base_url`, `api_key`, `ip_address`, or `port`
- `prompt_info`

### Initialization behavior

During construction the pipe:

1. builds one client for instruction parsing
2. builds one client for visual grounding
3. selects `google_style` or `openai_style` prompt serialization from the parser backend
4. resolves optional output schemas from prompt metadata

</div>

<div class="doc-api-block doc-api-block--methods">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--methods">Methods</span></div>

## Public Methods

| Method | Purpose |
| --- | --- |
| `_get_schema(schema_name=None)` | Resolve a schema class from `schemas.pipeline_io`. |
| `_check_format_by_schema(raw_data, schema_class)` | Validate model output against the resolved schema. |
| `_prepare_grounding_inputs(input_dict, objects_dict, edit_task_type)` | Decide which image or images to ground and which labels belong to each image. |
| `_valied_grounding_output(grounding_output, object_list)` | Verify that grounding output is a list and only contains expected labels. |
| `__call__(input_dict, **kwargs)` | Run the full parse-then-ground pipeline and return structured objects plus flattened coordinates. |

</div>

<div class="doc-api-block doc-api-block--signature">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--signature">Signature</span></div>

## Call Signature

```python
ParserGrounderPipe.__call__(
    input_dict: Dict[str, Any],
    **kwargs,
)
```

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Runtime Input Contract

### Common input fields

| Field | Required | Meaning |
| --- | --- | --- |
| `instruction` | Yes | Natural-language edit instruction. |
| `edit_task` | Yes | Normalized task type used to route grounding behavior. |
| `input_image` | Usually | Reference image before editing. |
| `edited_image` | Task-dependent | Edited image after applying the instruction. |

### Task routing behavior

`_prepare_grounding_inputs(...)` is where task-specific routing happens.

| `edit_task` family | Images sent to grounding | Labels used for grounding |
| --- | --- | --- |
| `SUBJECT_ADD` | `edited_image` | `edited_objects` |
| `SUBJECT_REMOVE`, `COLOR_ALTER`, `MATERIAL_ALTER` | `input_image` | `edited_objects` |
| `SUBJECT_REPLACE` | `input_image`, `edited_image` | `edited_objects`, `generated_objects` |
| `OBJECT_EXTRACTION`, `OREF`, `SIZE_ADJUSTMENT` | `input_image`, `edited_image` | `edited_objects` on both sides |
| `PS_HUMAN`, `MOTION_CHANGE` | `input_image`, `edited_image` | `edited_subjects` on both sides |
| `CREF` | `input_image` | `edited_subjects` |

If you add a new task family, this routing table is one of the first places that must change.

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Return Value

On success, the pipe returns:

```python
(
    extraction_json_response,
    all_coords,
)
```

### `extraction_json_response`

A parsed object dictionary returned by the instruction parser, augmented with:

- `bboxes`
  a list of bbox lists, one list per grounded image

Typical keys include:

- `edited_objects`
- `generated_objects`
- `edited_subjects`
- `edit_attributes`

depending on the prompt schema that was used.

### `all_coords`

A flattened list of `(x1, y1, x2, y2)` tuples in image pixel coordinates. Downstream modules often consume this directly.

</div>

<div class="doc-api-block doc-api-block--contract">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--contract">Input / Output</span></div>

## Coordinate Ordering Semantics

The coordinate order is not arbitrary. For multi-image tasks, the code flattens coordinates in the same order that images were processed. Several downstream metrics assume that:

- the first half belongs to the reference-side image
- the second half belongs to the edited-side image

This convention is especially important for:

- `sam_clip_cls_sim`
- `sam_dino_cls_sim`

If you change the ordering logic, you can silently break object-to-object pairing downstream.

</div>

<div class="doc-api-block doc-api-block--config">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--config">Config</span></div>

## Minimal Config Example

```yaml
parser_grounder_config:
  instruction_parser:
    default_config: ${pipes_default.instruction_parser}
    init_config:
      backend: vllm
      model_name: Qwen3-4B-Instruct-2507
      prompt_info:
        prompt_id: llm/instruction_parsing/basic_object
        version: v1

  general_grounder:
    default_config: ${pipes_default.general_grounder}
    init_config:
      backend: vllm
      model_name: Qwen3-VL-8B-Instruct
      prompt_info:
        prompt_id: vlm/grounding/general_grounding
        version: v1
```

This matches the current object-centric pipeline configs under `configs/pipelines/object_centric/`.

</div>

<div class="doc-api-block doc-api-block--failure">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--failure">Failure Mode</span></div>

## Failure and Retry Behavior

The pipe is intentionally retry-heavy because both parsing and grounding are prompt-driven.

### Instruction parsing failure

If the parser response:

- cannot be parsed as JSON, or
- does not satisfy the expected output schema

the pipe retries until `instruction_parser.retries` is exhausted.

### Grounding failure

Grounding is considered invalid when:

- the grounding JSON is not a list
- a returned label is not present in the expected object list
- no valid pixel coordinates can be extracted
- one of the grounded image slots ends up with an empty bbox list

If grounding still fails after the retry budget, the pipe returns:

```python
(None, None)
```

</div>

<div class="doc-api-block doc-api-block--extension">
<div class="doc-api-label-row"><span class="doc-api-label doc-api-label--extension">Extension</span></div>

## Extension Notes

- Add a new task family by updating `_prepare_grounding_inputs(...)` first.
- Keep prompt schemas aligned with `schemas.pipeline_io`, otherwise retries will mask real incompatibilities.
- Reuse this pipe if your change is only a new parsing prompt or grounding prompt.
- Create a new pipe only if the runtime is no longer "parse instruction, then ground visual targets."

</div>

</div>
