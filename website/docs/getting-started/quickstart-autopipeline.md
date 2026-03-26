# Quickstart

The goal of this page is not to explain every internal detail. It is to show the shortest path to the three primary AutoPipeline entry points:

1. `annotation`
2. `eval`
3. `train-pairs`

The commands below assume you have already completed [Installation](/docs/getting-started/environment-setup).

## CLI entry point

Preferred form:

```bash
autopipeline --help
```

If you have not installed the wrapper command, you can call the module directly:

```bash
python -m src.cli.autopipeline --help
```

## 1. Run annotation once

```bash
cd <PROJECT_ROOT>
autopipeline annotation \
  --edit-task subject_add \
  --pipeline-config-path <PROJECT_ROOT>/configs/pipelines/object_centric/subject_add.yaml \
  --user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml \
  --save-path <PROJECT_ROOT>/data/c_annotated_group_data
```

This command does the following:

- Resolves the candidate pool from `--edit-task`
- Loads the matching pipeline config
- Scores each candidate edited image with the configured metrics
- Aggregates candidates under the same source image and prompt into grouped JSONL

## 2. Run evaluation once

```bash
cd <PROJECT_ROOT>
autopipeline eval \
  --bmk geditv2 \
  --pipeline-config-path <PROJECT_ROOT>/configs/pipelines/vlm_as_a_judge/openai.yaml \
  --user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml \
  --save-path <PROJECT_ROOT>/data/reward_eval_results
```

This command is intended for benchmark comparison. It will:

- Read the benchmark definition from `configs/datasets/bmk.json`
- Form candidate image pairs
- Run the configured VLM-as-a-judge pipeline and emit normalized winners

:::note
Whether `eval` runs successfully depends on benchmark paths, model endpoints, and API credentials being available locally. The command format is stable, but the backing resources are not bundled with the docs.
:::

## 3. Convert grouped results into train pairs

```bash
cd <PROJECT_ROOT>
autopipeline train-pairs \
  --tasks subject_add \
  --input-dir <PROJECT_ROOT>/data/c_annotated_group_data \
  --output-dir <PROJECT_ROOT>/data/d_train_data \
  --mode auto
```

This command inspects the input schema and automatically selects:

- `group` mode for grouped annotation outputs
- `judge` mode for pairwise judge outputs

The result is a JSON file that can be passed directly into a downstream preference-training workflow.

## Expected artifacts

| Command | Key artifact |
| --- | --- |
| `annotation` | `<save-path>/<task>_grouped.jsonl` |
| `eval` | `<save-path>/<bmk>/<config-name>/<timestamp>.jsonl` |
| `train-pairs` | `<output-dir>/<task>.json` |
| All workflows | `<save-path>/.cache/*.jsonl` or the corresponding output directory |

## Where to go next

- If you want a deeper annotation walkthrough, read [First Annotation](/docs/tutorials/first-annotation)
- If you want benchmark judging, read [First Eval](/docs/tutorials/first-eval)
- If you want preference-data construction, read [First Train Pairs](/docs/tutorials/first-train-pairs)
- If you are unsure which pipeline family to use, read [How to Choose a Pipeline](/docs/tutorial/guide-for-pipelines/overview)
