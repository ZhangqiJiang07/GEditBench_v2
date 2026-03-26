# Framework Overview

AutoPipeline is not a loose collection of scripts. It is a runtime organized around configuration, normalized inputs, and reusable output artifacts.

From the codebase perspective, the core runtime path is:

```text
CLI
  -> ConfigEngine
  -> Runner / Worker / CacheManager
  -> PipelineLoader(name)
  -> Pipeline
      -> ParserGrounder
      -> Metric Pipes / Judge Pipes
  -> grouped results / eval results / train pairs
```

## System layers

| Layer | Responsibility | Code location |
| --- | --- | --- |
| CLI | Unified user entry point | `src/cli/autopipeline.py` |
| Config Engine | Resolves `_base_`, variable references, and merged defaults | `src/core/config_engine.py` |
| Runner | Owns dataset loading, execution, cache, and the main loop | `src/autopipeline/runners/` |
| Pipelines | Defines workflow-specific input preparation and orchestration | `src/autopipeline/pipelines/` |
| Modules | Registered metric and judge execution units | `src/autopipeline/components/modules/` |
| Primitives | Lower-level experts, clients, prompt adapters, and vision mixins | `src/autopipeline/components/primitives/` |
| Postprocess | Converts grouped results into training pairs | `src/autopipeline/postprocess/` |

## Pipeline families

### `object-centric`

Use this family for questions such as:

- Was the target object correctly added, removed, or replaced?
- Did the unedited region remain stable?
- Did text edits, material swaps, or size changes behave as expected?

### `human-centric`

Use this family for questions such as:

- Was facial identity, geometry, hair, or body appearance preserved?
- Does the edit affect a local region or the overall human subject?
- Do you need separate measurements for the edit area and the unedited area?

### `vlm-as-a-judge`

Use this family for questions such as:

- Which of two candidate edits is better?
- Which model wins on a benchmark?
- Do you want pairwise preference data directly from the evaluation path?

## Why this structure works for data construction

The framework separates three concerns cleanly:

- dataset loading
- evaluation logic
- downstream artifact formats

That separation lets you evolve the system without rewriting the CLI:

- add a new task config
- replace a metric pipe
- swap a prompt template
- route grouped results into a new post-processing stage

## Two concepts every new user should understand first

### 1. Configuration comes before code

Most routine usage does not require Python changes. In practice, you usually only need to:

- choose a pipeline family
- start from an existing config
- update `user_config.yaml`
- run the CLI

### 2. Output artifacts are the downstream interface

In AutoPipeline, the most important interface is not the class name but the file artifact:

- grouped JSONL
- eval JSONL
- train pair JSON

That is why the docs treat output formats as a first-class topic.

If you want the internal extension surface next, continue with [Components Overview](/docs/components/overview). If you want the configuration model first, read [Config System](/docs/concepts/config-system).
