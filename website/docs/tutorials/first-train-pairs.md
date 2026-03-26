# First Train Pairs

`train-pairs` is AutoPipeline's post-processing entry point. It converts intermediate artifacts into pairwise data that can be consumed by a training pipeline.

## Accepted inputs

Two input families are supported:

- grouped results produced by `annotation`
- pairwise grouped results produced by `eval` or another judge-based workflow

The CLI inspects the `results` field and automatically detects:

- `group` mode
- `judge` mode

## Minimal command

```bash
cd <PROJECT_ROOT>
autopipeline train-pairs \
  --tasks subject_add \
  --input-dir <PROJECT_ROOT>/data/c_annotated_group_data \
  --output-dir <PROJECT_ROOT>/data/d_train_data \
  --mode auto
```

## Common parameters

- `--tasks`
  Comma-separated task list. Multiple tasks can be processed in one run.
- `--prompts-num`
  Maximum number of prompts or groups to process per task.
- `--mode`
  `auto`, `group`, or `judge`.
- `--filt-out-strategy`
  `head_tail` or `three_tiers`. Only applies to `group` mode.
- `--thresholds-config-file`
  Threshold config used for tiering and filtering in `group` mode.

## Output files

Each task produces one JSON file:

```text
<output-dir>/<task>.json
```

## Simplified output example

```json
[
  {
    "edited_image_paths": [
      "images/edited/model_a/0001.png",
      "images/edited/model_b/0001.png"
    ],
    "instruction": "Add a red balloon to the left side of the boy.",
    "source_image_path": "images/source/0001.png",
    "gpt_response": "```json\n{\n    \"winner\": \"Image A\"\n}\n```"
  }
]
```

## `group` mode vs `judge` mode

### `group` mode

This mode consumes grouped annotation results from human-centric or object-centric pipelines.

The conversion logic will:

- clean the candidate set
- compute z-scores
- assign tiers according to area-specific configs
- generate chosen/rejected pairs

### `judge` mode

This mode consumes pairwise judge outputs.

It reads the winner directly and converts each judged pair into a training sample.

## When to use it

If your goal is not just to inspect evaluation artifacts but to continue into training, this is the correct next step.
