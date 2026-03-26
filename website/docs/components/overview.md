# Components Overview

The `components/` layer is where AutoPipeline's reusable execution units live. It is the bridge between high-level pipeline YAML and the actual logic that runs on images, prompts, masks, and model backends.

At a practical level, `components/` is split into two parts:

- `modules/`
  Registered execution units that are selected directly by pipeline configs.
- `primitives/`
  Lower-level clients, experts, prompt adapters, mixins, and utilities that modules build on.

<div class="doc-kind-row"><span class="doc-kind doc-kind--overview">Overview</span></div>

<div class="doc-section-card">

## The two-layer model

| Layer | Purpose | Typical config surface | Examples |
| --- | --- | --- | --- |
| `modules/` | Execute a metric, a judge, or parsing/grounding logic | `metric_configs[*].pipe_name`, `init_config`, `runtime_params` | `lpips-pipe`, `pairwise-judge`, `parser-grounder` |
| `primitives/` | Provide reusable lower-level capabilities | nested `init_config`, `expert_configs`, client settings, model paths | `api` client, `openai_style` prompt adapter, `face-detector` expert |

## Registry surfaces

AutoPipeline relies on registry-backed extension points throughout this layer:

- `PIPE_REGISTRY`
  Registers executable modules in `components/modules/`.
- `CLIENT_REGISTRY`
  Registers backend clients such as `api`, `google`, and `vllm`.
- `PROMPT_ADAPTER_REGISTRY`
  Registers payload builders such as `openai_style` and `google_style`.
- `EXPERT_REGISTRY`
  Registers human-centric experts such as `face-detector`, `human-segmenter`, and `hair-segmenter`.

One important implementation detail: registration is import-side-effect driven. A new file is not part of the framework surface until it is imported by the appropriate `__init__.py`.

</div>

<div class="doc-section-card">

## How the component layer is used at runtime

The runtime path is:

```text
pipeline YAML
  -> metric_configs / parser_grounder_config / expert_configs
  -> registry lookup
  -> module instantiation
  -> primitive-backed execution
  -> score, winner, or parsed region artifact
```

In practice:

- `metric_configs` chooses a registered module by `pipe_name`
- `scope` becomes `mask_mode` in the pipeline runtime
- `runtime_params` are passed back into the module call
- `expert_configs` are loaded only when a human-centric metric family needs them

</div>

<div class="doc-section-card">

## Recommended reading order

If you only need to run the framework, you can skip most of this section.

If you need to extend the framework, read in this order:

1. [Modules Overview](/docs/components/modules/overview)
2. [ParserGrounderPipe](/docs/components/modules/parser-grounder-pipe)
3. [Judge Modules](/docs/components/modules/judge-modules)
4. [Primitives Overview](/docs/components/primitives/overview)
5. [BaseClient](/docs/components/primitives/clients/base-client)
6. [MaskProcessor](/docs/components/primitives/mask-processor)

</div>

<div class="doc-section-card">

## What belongs in configs vs code

As a rule:

- choose or combine existing modules in YAML whenever possible
- add a new primitive only when you need a new reusable low-level capability
- add a new module when the framework needs a new executable metric, judge, or parsing unit

If your goal is only to add a new task, you usually do not need to edit `components/` at all.

</div>

<div class="doc-section-card">

## Source-aligned documentation structure

The component docs are intentionally aligned with the source tree instead of being grouped into broad topic pages.

That means:

- module docs are split by implementation file or by a very tight functional grouping
- primitive docs are split by source file, especially for clients, analyzers, and reusable mixins
- closely related classes can still share one page when they form a single runtime surface

This structure makes it easier to compare documentation with `src/autopipeline/components/` while you read the code.

</div>
