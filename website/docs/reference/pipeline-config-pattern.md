# How to Add a Pipeline Config

This page assumes you are not trying to redesign the runtime. The goal is to add a new task config with minimal Python changes.

## Decide which layer you actually need to change

- If you are only adding a task, a pipeline YAML is usually enough
- If you are only switching models or prompts, update `init_config` or `prompt_info`
- If you need a new metric, you may need a new module or a new mapping
- If you need a new input schema, you may also need a new prompt asset and schema class

## Minimal template

```yaml
_base_:
  pipes_default: "../modules_init/pipes_default.yaml"
  experts_default: "../modules_init/experts_default.yaml"

name: object-centric
support_task:
  - subject_add

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

metric_configs:
  lpips:
    pipe_name: lpips-pipe
    default_config: ${pipes_default.lpips-pipe}
    init_config:
    scope: unedit_area
```

## Recommended workflow

1. Copy the closest starter config
2. Make sure `name` is an existing registered pipeline name
3. Update `support_task` so it matches the CLI `--edit-task`
4. Update `prompt_info` and confirm the prompt asset exists
5. Check each `pipe_name` and `scope` in `metric_configs`
6. If the pipeline is human-centric, complete `expert_configs`
7. Validate with a small annotation run before scaling up

## When YAML is no longer enough

Python changes are usually necessary only when:

- no existing module can express the metric you need
- the prompt output schema requires new fields
- the dataset loader cannot consume your data format
- the post-processing logic needs a genuinely new conversion path

## Configuration checklist

- Does `support_task` include the task you will pass to the CLI?
- Does `prompt_id/version` point to a real prompt asset?
- Is every `pipe_name` already registered?
- Do `model_paths` and `client_config` resolve on the current machine?
- Are all required human-centric experts configured?
