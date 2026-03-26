# Extending AutoPipeline

AutoPipeline already exposes several extension surfaces, but they are not all equivalent in cost.

In practice, most changes should stop at one of these three layers:

- a new `primitive`
- a new `module` or `pipe`
- a new `pipeline` family

<div class="doc-kind-row"><span class="doc-kind doc-kind--overview">Guide</span></div>

<div class="doc-section-card">

## Choose the Smallest Extension Surface

| If you need to... | Preferred extension point | Typical files |
| --- | --- | --- |
| Reuse a low-level backend, detector, segmenter, or helper across multiple pipes | Primitive | `src/autopipeline/components/primitives/` |
| Add a new executable metric, judge, or parsing block referenced by `pipe_name` | Module / Pipe | `src/autopipeline/components/modules/` |
| Add a new runtime orchestration pattern with its own input shaping and execution flow | Pipeline family | `src/autopipeline/pipelines/` |

The practical rule is simple:

- if YAML composition is enough, do not add a new Python type
- if several metrics need the same low-level capability, add a primitive
- if one new score or judge behavior must run from `metric_configs`, add a pipe
- if the runtime no longer fits `object-centric`, `human-centric`, or `vlm-as-a-judge`, add a new pipeline family

</div>

<div class="doc-section-card">

## What Is Already Registry-Backed

AutoPipeline relies on import-side-effect registration in several places:

- `PIPE_REGISTRY` for executable pipes
- `EXPERT_REGISTRY` for expert-style primitives
- `CLIENT_REGISTRY` for backend clients
- `PROMPT_ADAPTER_REGISTRY` for prompt serialization
- `PIPELINE_REGISTRY` for top-level pipeline families

That means a new class is not really part of the framework until the corresponding package `__init__.py` imports it.

</div>

<div class="doc-section-card">

## Recommended Reading Order for Extension Work

1. Read [Components Overview](/docs/components/overview) to understand the runtime split between modules and primitives.
2. Read [Modules Overview](/docs/components/modules/overview) and [Primitives Overview](/docs/components/primitives/overview) to find an existing pattern close to your use case.
3. Start with one of these focused guides:
   - [Add a Primitive](/docs/extending/add-primitive)
   - [Add a Module or Pipe](/docs/extending/add-module)
   - [Add a Pipeline Family](/docs/extending/add-pipeline)

</div>

<div class="doc-section-card">

## Before You Add a New Type

The current codebase already contains a few important reusable pieces:

- `BasePipeline` handles config validation, parser-grounder loading, expert loading, image parsing, and metric pipe loading.
- `parser-grounder` is already the standard front end for region-aware pipelines.
- one pipe class can expose multiple metrics because the YAML metric key and the registered `pipe_name` are different surfaces.
- pipes with the same `(pipe_name, init_config)` share one cached instance inside a pipeline runtime.

Those constraints mean the cheapest successful extension is often:

1. reuse an existing primitive
2. add one new metric branch in an existing pipe
3. add one new pipeline YAML

Only go wider when that stops being structurally correct.

</div>

<div class="doc-section-card">

## Current Framework Caveats

A few extension points are more hardcoded than they first appear:

- `parser-grounder` is loaded explicitly by the existing region-aware pipeline families rather than through `metric_configs`
- `BasePipeline` only maps `edit_area` and `unedit_area` into `mask_mode`
- human-centric expert loading is coupled to metric-name prefixes such as `face_*`, `hair_*`, and `body_*`
- the runner and worker layer still contains explicit branching on the current pipeline families

This is why adding a primitive or a pipe is usually straightforward, while adding a new pipeline family requires more integration work.

</div>

<div class="doc-section-card">

## Verification Strategy

No matter which layer you extend, validate in this order:

1. import and registration
2. one minimal config file
3. one single-sample end-to-end run
4. output artifact shape
5. only then multi-worker execution

That order will save time because most real failures happen in registration, config merge, prompt assets, or runtime input-shape mismatches.

</div>
