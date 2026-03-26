# CLI Reference

The unified AutoPipeline entry point is:

```bash
autopipeline
```

If you have not installed the wrapper command, you can call the module directly:

```bash
python -m src.cli.autopipeline
```

## Subcommands

- `annotation`
- `eval`
- `train-pairs`

:::note
The current implementation hardcodes many defaults under `<PROJECT_ROOT>/...`. If your checkout lives elsewhere, pass `--pipeline-config-path`, `--user-config`, `--save-path`, and related paths explicitly.
:::

## `annotation`

Purpose: run a human-centric or object-centric pipeline and score candidate edited images with structured metrics.

```bash
autopipeline annotation \
  --edit-task <task> \
  --pipeline-config-path <pipeline-yaml> \
  [--max-workers 4] \
  [--save-path <PROJECT_ROOT>/data/c_annotated_group_data] \
  [--user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml] \
  [--candidate-pool-dir <PROJECT_ROOT>/configs/datasets/candidate_pools]
```

Parameters:

| Parameter | Required | Description |
| --- | --- | --- |
| `--edit-task` | Yes | Edit task name. The CLI normalizes it into lowercase underscore form. |
| `--pipeline-config-path` | Yes | Absolute path to the pipeline YAML. |
| `--max-workers` | No | Worker parallelism. |
| `--save-path` | No | Output directory for results and cache. |
| `--user-config` | No | User config YAML. |
| `--candidate-pool-dir` | No | Directory containing candidate pool JSON files. |

## `eval`

Purpose: run `vlm-as-a-judge` and produce pairwise winners on a benchmark.

```bash
autopipeline eval \
  --bmk <benchmark-key> \
  --pipeline-config-path <pipeline-yaml> \
  [--max-workers 4] \
  [--save-path <PROJECT_ROOT>/data/reward_eval_results] \
  [--user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml] \
  [--bmk-config <PROJECT_ROOT>/configs/datasets/bmk.json] \
  [--geditv2-metadata-file metadata.jsonl]
```

Parameters:

| Parameter | Required | Description |
| --- | --- | --- |
| `--bmk` | Yes | Benchmark key defined in `bmk.json`. |
| `--pipeline-config-path` | Yes | Judge pipeline YAML. |
| `--max-workers` | No | Worker parallelism. |
| `--save-path` | No | Result directory. |
| `--user-config` | No | User config YAML. |
| `--bmk-config` | No | Benchmark config file. |
| `--geditv2-metadata-file` | No | Metadata filename used only for `geditv2` evaluation. |

## `train-pairs`

Purpose: convert grouped results into preference-training data.

```bash
autopipeline train-pairs \
  --tasks <task1,task2,...> \
  [--prompts-num 1500] \
  [--prefix ""] \
  [--input-dir <PROJECT_ROOT>/data/c_annotated_group_data] \
  [--output-dir <PROJECT_ROOT>/data/d_train_data] \
  [--mode auto] \
  [--filt-out-strategy three_tiers] \
  [--thresholds-config-file <PROJECT_ROOT>/configs/pipelines/data_construction_configs.json]
```

Parameters:

| Parameter | Required | Description |
| --- | --- | --- |
| `--tasks` | Yes | Comma-separated task names. |
| `--prompts-num` | No | Maximum number of prompt groups per task. |
| `--prefix` | No | Optional output subdirectory prefix. |
| `--input-dir` | No | Input directory for grouped results. |
| `--output-dir` | No | Output directory for train pairs. |
| `--mode` | No | `auto`, `group`, or `judge`. |
| `--filt-out-strategy` | No | `head_tail` or `three_tiers`. |
| `--thresholds-config-file` | No | Threshold config for `group` mode. |

## Two common invocation styles

### User-facing CLI

```bash
autopipeline annotation ...
autopipeline eval ...
autopipeline train-pairs ...
```

### Module entry point

```bash
python -m src.cli.autopipeline annotation ...
python -m src.cli.autopipeline eval ...
python -m src.cli.autopipeline train-pairs ...
```
