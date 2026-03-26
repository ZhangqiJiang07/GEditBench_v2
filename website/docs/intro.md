---
sidebar_position: 1
---

# AutoPipeline

AutoPipeline is an open-source framework for image-edit evaluation and data construction. It unifies three practical workflows, `annotation`, `eval`, and `train-pairs`, behind a single CLI, a shared configuration system, and consistent output artifacts.

If you came from the site homepage, this is the actual documentation-oriented starting point. The homepage is reserved for the paper and benchmark presentation, while this page is the product and runtime overview.

These docs are written for users who need to answer questions such as:

- How do I score a batch of edited images and keep only the strongest candidates?
- How do I run pairwise judging on a benchmark and compare model outputs?
- How do I turn intermediate artifacts into preference-training data instead of writing one-off scripts?
- How do I add a task config or metric config without reading the entire codebase first?

## How to read these docs

Recommended reading order:

1. [Installation](/docs/getting-started/environment-setup)
2. [Quickstart](/docs/getting-started/quickstart-autopipeline)
3. [First Annotation](/docs/tutorials/first-annotation)
4. [First Eval](/docs/tutorials/first-eval)
5. [First Train Pairs](/docs/tutorials/first-train-pairs)
6. [How to Choose a Pipeline](/docs/tutorial/guide-for-pipelines/overview)
7. [Components Overview](/docs/components/overview)
8. [Extending AutoPipeline](/docs/extending/overview)

## What you can do with AutoPipeline

| Workflow | What it does | Primary artifact |
| --- | --- | --- |
| `annotation` | Score candidate edited images with structured metrics | `<save-path>/<task>_grouped.jsonl` |
| `eval` | Run pairwise judging on a benchmark | `<save-path>/<bmk>/<config>/<timestamp>.jsonl` |
| `train-pairs` | Convert grouped results into preference-training data | `<output-dir>/<task>.json` |

## Runtime flow

The core runtime can be understood in five steps:

1. The CLI normalizes your task, benchmark, and path inputs.
2. `ConfigEngine` resolves `_base_`, `${...}` references, and merged defaults.
3. The runner loads the dataset, executor, and cache strategy.
4. The runtime dispatches to one pipeline family and the required modules.
5. The workflow emits grouped scores, winners, or training pairs.

## Pipeline families

### `object-centric`

Use this family for object-level edits such as addition, removal, replacement, color modification, material modification, size adjustment, or text editing.

### `human-centric`

Use this family when evaluation depends on face, hair, body pose, or body appearance consistency.

### `vlm-as-a-judge`

Use this family when the core task is pairwise comparison rather than structured scoring for a single image.

## The output artifacts are the real interface

In day-to-day usage, the most important interface is not a Python class name but the artifact emitted by the workflow:

- grouped JSONL for annotation
- eval JSONL for pairwise benchmark judging
- task-level JSON for training pairs

This is why the docs treat output formats as a first-class topic rather than a minor appendix.

## The most important layers in the codebase

- `src/cli/autopipeline.py`
  Unified CLI entry point.
- `src/core/config_engine.py`
  Resolves `_base_`, `${...}` references, and merged defaults.
- `src/autopipeline/pipelines/`
  The three pipeline families.
- `src/autopipeline/components/`
  Registered modules and lower-level primitives.
- `src/autopipeline/postprocess/train_pairs.py`
  Conversion logic from grouped results to training pairs.

## Where to go next

- If you want to run the shortest successful command path, go to [Quickstart](/docs/getting-started/quickstart-autopipeline)
- If you need to choose the correct runtime family, go to [How to Choose a Pipeline](/docs/tutorial/guide-for-pipelines/overview)
- If you need to understand the extension surface, go to [Components Overview](/docs/components/overview)
- If you want to add a new primitive, pipe, or pipeline family, go to [Extending AutoPipeline](/docs/extending/overview)

:::tip
This documentation is intentionally workflow-first. Run one complete command successfully before you dive into the internal component pages; the architecture will make far more sense once you have seen the artifacts it produces.
:::
