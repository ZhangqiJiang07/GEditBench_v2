# Object-Centric Pipeline

`object-centric` is the easiest pipeline family to start with. It is designed for object-level evaluation of a single edited image and focuses on:

- whether the edit target can be parsed and localized correctly
- whether the unedited region remains stable
- whether structure and semantics remain plausible for the task

## Supported starter configs

The repository already includes starter configs such as:

- `subject_add.yaml`
- `subject_remove.yaml`
- `subject_replace.yaml`
- `color_alter.yaml`
- `material_alter.yaml`
- `size_adjustment.yaml`
- `text_editing.yaml`
- `cref.yaml`
- `oref.yaml`

## What you need before running it

- an object-centric pipeline YAML
- a valid `user_config.yaml`
- the candidate pool JSON for the task

`annotation` resolves the candidate pool automatically from `--edit-task`, so the task name and the chosen config must match.

## Configuration shape

The two most important config blocks are:

- `parser_grounder_config`
- `metric_configs`

In other words, this family usually does not require `expert_configs`.

## Minimal example

```bash
cd <PROJECT_ROOT>
autopipeline annotation \
  --edit-task subject_add \
  --pipeline-config-path <PROJECT_ROOT>/configs/pipelines/object_centric/subject_add.yaml \
  --user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml \
  --save-path <PROJECT_ROOT>/data/c_annotated_group_data
```

## What it actually does

1. Normalizes `instruction`, `input_image`, and `edited_images[0]` into a common input schema
2. Uses parser-grounder to parse the instruction and localize relevant regions
3. Resolves each metric to a registered pipe from `metric_configs`
4. Computes scores for `edit_area`, `unedit_area`, or both, depending on the config
5. Aggregates results into grouped JSONL

## Good fit vs poor fit

Good fit:

- structured scoring of a single edited image
- object or scene-level edits
- metric-driven filtering and data construction

Poor fit:

- tasks dominated by facial identity or body-local consistency
- problems that are fundamentally pairwise comparison tasks

If your problem is primarily about human consistency, continue with [Human-Centric](/docs/tutorial/guide-for-pipelines/human-centric).
