# Config System

The core of AutoPipeline is not merely that it registers many classes. The real objective is to turn a pipeline YAML file into a stable, executable runtime configuration. The config system performs three jobs:

1. `_base_` inheritance
2. `${...}` variable resolution
3. `default_config` and `init_config` merging

## 1. `_base_`: attach default files as namespaces

A typical pattern looks like this:

```yaml
_base_:
  pipes_default: "../modules_init/pipes_default.yaml"
  experts_default: "../modules_init/experts_default.yaml"
```

This is not a plain include. Each referenced file is attached as a namespace that can be reused later:

```yaml
default_config: ${pipes_default.clip-pipe}
```

## 2. `${...}`: resolve variable references

Common patterns include:

```yaml
model_path: ${user_config.model_paths.clip_path}
api_key: ${client_config.api_key}
default_config: ${pipes_default.instruction_parser}
```

`ConfigEngine` resolves these references before pipeline instantiation.

## 3. `default_config` plus `init_config`

This is the most important merge step.

In a pipeline YAML, you will usually see a pattern like:

```yaml
metric_configs:
  emd:
    pipe_name: clip-pipe
    default_config: ${pipes_default.clip-pipe}
    init_config:
    scope: unedit_area
    runtime_params:
      patch_mask_threshold: 0.1
```

At load time, the system merges `default_config` and `init_config` into the effective initialization config. In practice, the clean rule is:

- put reusable defaults into `modules_init/*.yaml`
- put task-specific overrides into the active pipeline YAML

## Files users modify most often

### `configs/pipelines/user_config.yaml`

This is the file most users edit first. It controls:

- `client_config`
- `model_paths`

In many cases, updating this file is enough to adapt the runtime to your own machine.

### `configs/pipelines/modules_init/pipes_default.yaml`

This file holds reusable pipe defaults, for example:

- CLIP model paths
- DINO model paths
- default VLM and LLM backend configs

### `configs/pipelines/modules_init/experts_default.yaml`

This file holds reusable expert defaults, for example:

- face detector
- hair segmenter
- human segmenter

## How prompt assets are connected

Many client-backed configs include:

```yaml
prompt_info:
  prompt_id: llm/instruction_parsing/basic_object
  version: v1
```

At runtime, `PromptAssetStore` resolves this entry against the YAML prompt assets in `src/prompts/assets/`.

When you add a new prompt-driven component, you must make sure that:

1. `prompt_id` and `version` point to a real asset
2. the output schema matches the prompt semantics

## Recommended configuration workflow

1. Choose the pipeline family
2. Copy the closest starter config
3. Update `support_task`
4. Update `prompt_info`
5. Update `metric_configs`
6. Only then consider Python changes

If you want to author a new task config, continue with [How to Add a Pipeline Config](/docs/reference/pipeline-config-pattern).
