# GEditBench v2

GEditBench v2 is an end-to-end framework for building, evaluating, and training visual consistency judges for image editing.

It connects three stages that are usually scattered across separate research codebases: automatic edited-candidate generation (`autogen`), automatic visual-consistency preference construction (`autopipeline`), and LoRA-based VLM training (`autotrain`). The repository also includes reusable benchmark/evaluation utilities, prompt assets, and lightweight CLIs for running the full workflow from data construction to judge inference.

This repository accompanies an upcoming paper. Citation details will be added once the manuscript is public.

> Important
> 
> The current codebase still reflects internal experiment layouts in several places. Before running it in a new environment, update local model paths, dataset roots, benchmark roots, and API/vLLM endpoints in `configs/`, `src/autogen/constants.py`, and related scripts.

## Highlights

- End-to-end workflow from raw image-instruction pairs to a trained pairwise VLM judge.
- Automatic candidate generation for edited images and benchmark generation for GEditBench v2-style evaluation.
- Automatic visual-consistency annotation with object-centric, human-centric, and VLM-as-a-judge pipelines.
- Preference-pair construction utilities for turning grouped annotations into training data.
- LoRA-based training scripts for Qwen-family VLM judges, with DeepSpeed support.
- Config-driven design with reusable prompts, modular pipelines, and publishable environment profiles.
- Lightweight CLIs for generation, annotation, pair construction, training, and evaluation.

## What This Repo Includes

- `autogen`
  Generates edited candidates from image-editing models, filters raw image-prompt pairs with Qwen3-VL embeddings, and prepares GEditBench v2 benchmark outputs.

- `autopipeline`
  Builds visual-consistency annotations and preference pairs with modular pipelines. It supports object-centric checks, human-centric checks, and VLM-as-a-judge evaluation.

- `autotrain`
  Trains pairwise VLM judges with LoRA/DoRA-style adaptation and DeepSpeed-based launch scripts.

- `inference`
  Runs local evaluation and benchmark inference for trained LoRA judges.

- `configs`, `prompts`, and `environments`
  Provide experiment configs, prompt assets, and environment profiles for reproducible runs.

## End-to-End Workflow

```text
Raw image-prompt pairs
        |
        v
autogen filter
        |
        v
autogen candidate generation
        |
        v
autopipeline annotation
        |
        v
autopipeline train-pairs
        |
        v
autotrain LoRA training
        |
        v
local inference / benchmark evaluation
```

In practical terms, the repository supports the following loop:

1. Sample or filter diverse source image-instruction pairs.
2. Generate multiple edited candidates with image-editing models.
3. Annotate visual consistency automatically with task-specific pipelines.
4. Convert grouped results into pairwise preference data.
5. Train a VLM judge on those pairs.
6. Evaluate the resulting judge on GEditBench v2 or reward benchmarks.

## Installation

### Requirements

- Linux with Python `3.11` or `3.12`
- CUDA-capable GPU(s) for candidate generation and training
- Local or remote access to required model checkpoints
- Optional API or vLLM endpoints for judge-based pipelines

### Environment Profiles

The repository ships with three environment profiles:

| Profile | Purpose |
| --- | --- |
| `annotate` | `autogen` and `autopipeline` workflows |
| `train` | `autotrain` LoRA training |
| `pvc_judge` | local pairwise-judge inference and evaluation |

Environment exports and lock files live under [`environments/`](./environments). See [`environments/README.md`](./environments/README.md) for the full environment-management details.

### Recommended Setup

Install the environment profile you need for the current stage:

```bash
# For autogen + autopipeline
./scripts/install.sh annotate
source .venvs/annotate/bin/activate

# For training
./scripts/install.sh train
source .venvs/train/bin/activate

# For local judge inference
./scripts/install.sh pvc_judge
source .venvs/pvc_judge/bin/activate
```

During development, you can invoke the CLIs directly with `python -m src.cli.<tool>`. Optional wrapper installers are also provided:

```bash
./scripts/install_autogen.sh
./scripts/install_autopipeline.sh
./scripts/install_autotrain.sh
```

### Configuration You Should Update First

Before the first run, review these files and replace internal defaults with your own paths and endpoints:

- `configs/pipelines/user_config.yaml`
- `configs/datasets/bmk.json`
- `configs/lora_sft/*.yaml`
- `src/autogen/constants.py`

At minimum, check:

- model checkpoint roots
- benchmark/data roots
- vLLM or API endpoints
- credentials and API keys
- output directories

## Quick Start

The examples below use direct module invocation so they work even before installing shell wrappers.

### 1. Filter raw image-instruction pairs

```bash
python -m src.cli.autogen filter \
  --task subject_add \
  --input-file /path/to/raw_pairs.jsonl \
  --output-dir $(pwd)/data/b_filtered_img_prompt_pair_data \
  --qwen-embedding-model-path /path/to/Qwen3-VL-Embedding-8B
```

This stage selects a diverse subset of source examples and writes a task-specific `meta_info.jsonl` under `data/b_filtered_img_prompt_pair_data/<task>/`.

### 2. Generate edited candidates

```bash
python -m src.cli.autogen run candidates \
  --task subject-add \
  --model qwen-image-edit \
  --dataset-path $(pwd)/data/b_filtered_img_prompt_pair_data \
  --gpus-per-worker 1
```

Supported generation backends in the current codebase include `qwen-image-edit`, `qwen-image-edit-2509`, `qwen-image-edit-2511`, `step1x_edit1.2`, `step1x_edit1.2-preview`, `kontext`, and several GEditBench v2 benchmark generation models such as `flux.2_dev`, `glm_image`, `longcat_image_edit`, and `FireRed-Image-Edit-1.1`.

### 3. Run automatic annotation

```bash
python -m src.cli.autopipeline annotation \
  --edit-task subject_add \
  --pipeline-config-path $(pwd)/configs/pipelines/object_centric/subject_add.yaml \
  --save-path $(pwd)/data/c_annotated_group_data \
  --user-config $(pwd)/configs/pipelines/user_config.yaml \
  --candidate-pool-dir $(pwd)/configs/datasets/candidate_pools
```

This stage aggregates per-candidate results into grouped task outputs such as `data/c_annotated_group_data/subject_add_grouped.jsonl`.

### 4. Convert grouped results into preference pairs

```bash
python -m src.cli.autopipeline train-pairs \
  --tasks subject_add \
  --input-dir $(pwd)/data/c_annotated_group_data \
  --output-dir $(pwd)/data/d_train_data \
  --mode auto \
  --filt-out-strategy three_tiers \
  --thresholds-config-file $(pwd)/configs/pipelines/data_construction_configs.json
```

### 5. Train a LoRA judge

```bash
python -m src.cli.autotrain \
  --config qwen3_vl_8b_train \
  --config-path $(pwd)/configs/lora_sft \
  --num-gpus 8
```

The training launcher resolves the YAML config, creates an output directory, and starts DeepSpeed on `src/autotrain/train/train_sft_lora.py`.

## CLI Overview

GEditBench v2 exposes three primary CLIs:

| CLI | Scope | Representative Commands |
| --- | --- | --- |
| `autogen` | data filtering and candidate generation | `filter`, `run candidates`, `run geditv2` |
| `autopipeline` | annotation, evaluation, and pair construction | `annotation`, `eval`, `train-pairs` |
| `autotrain` | LoRA VLM training launcher | top-level training entry |

Examples:

```bash
python -m src.cli.autogen --help
python -m src.cli.autopipeline --help
python -m src.cli.autotrain --help
```

If you prefer shell commands instead of module invocation:

```bash
autogen --help
autopipeline --help
autotrain --help
```

## Supported Tasks And Pipelines

### Task Families

The current repository contains candidate-pool or pipeline configs for tasks such as:

- `background_change`
- `camera_motion`
- `color_alter`
- `cref`
- `enhancement`
- `material_alter`
- `motion_change`
- `object_extraction`
- `oref`
- `ps_human`
- `size_adjustment`
- `sref`
- `style_transfer`
- `subject_add`
- `subject_remove`
- `subject_replace`
- `text_editing`
- `tone_transfer`

### Pipeline Families

Available pipeline recipes are organized under `configs/pipelines/`:

- `object_centric`
  For tasks where local object appearance, structure, or identity are the primary concern.

- `human_centric`
  For tasks involving people, pose, hair, face identity, or body-consistency constraints.

- `vlm_as_a_judge`
  For pairwise or score-based VLM evaluation pipelines.

- `ablation`
  For model-comparison or judge-comparison experiments.

## Repository Structure

```text
GEditBench_v2/
├── configs/
│   ├── datasets/              # candidate pools, benchmark definitions
│   ├── lora_sft/              # LoRA/VLM training configs
│   └── pipelines/             # annotation and eval pipeline configs
├── data/
│   ├── a_raw_img_prompt_pair_data/
│   ├── b_filtered_img_prompt_pair_data/
│   ├── c_annotated_group_data/
│   ├── d_train_data/
│   ├── e_geditv2_pair_res/
│   ├── f_reward_results/
│   └── z_reward_bench/
├── environments/              # publishable env profiles and lock files
├── scripts/                   # installers and utility launchers
├── src/
│   ├── autogen/
│   ├── autopipeline/
│   ├── autotrain/
│   ├── cli/
│   ├── inference/
│   └── prompts/
└── vllm_deploy_scripts/       # helper scripts for serving judge backends
```

## Configuration And Data Convention

### Config-Driven Design

Most workflows are driven by explicit config files rather than hardcoded experiment logic:

- `configs/datasets/`
  Candidate-pool definitions and benchmark roots.

- `configs/pipelines/`
  Pipeline recipes for object-centric, human-centric, judge-based, and ablation experiments.

- `configs/lora_sft/`
  Training recipes for LoRA-based VLM fine-tuning.

- `src/prompts/assets/`
  Versioned prompt assets for LLM parsing, grounding, visual consistency, instruction following, and visual quality.

### Staged Data Layout

The `data/` directory is organized as a staged workflow:

| Stage | Meaning |
| --- | --- |
| `a_raw_img_prompt_pair_data` | raw source pairs before filtering |
| `b_filtered_img_prompt_pair_data` | filtered subsets and generated candidate metadata |
| `c_annotated_group_data` | grouped annotation outputs |
| `d_train_data` | pairwise training data for judge learning |
| `e_geditv2_pair_res` | GEditBench v2 evaluation metadata/results |
| `f_reward_results` | reward/judge evaluation outputs |
| `z_reward_bench` | benchmark assets and shuffled benchmark annotations |

This staged naming makes it easier to cache intermediate outputs, rerun only one phase, and keep experiment artifacts separate.

## Evaluation

There are two main evaluation paths in the current repository.

### Config-Driven Pipeline Evaluation

Use `autopipeline eval` when you want to run evaluation through a configured pipeline:

```bash
python -m src.cli.autopipeline eval \
  --bmk vc_reward \
  --pipeline-config-path $(pwd)/configs/pipelines/ablation/eval_vc_qwen3.yaml \
  --save-path $(pwd)/data/f_reward_results \
  --user-config $(pwd)/configs/pipelines/user_config.yaml \
  --bmk-config $(pwd)/configs/datasets/bmk.json
```

Current benchmark keys defined in `configs/datasets/bmk.json` include:

- `editscore_consistency`
- `editscore_prompt_following`
- `editreward_visual_quality`
- `vc_reward`
- `geditv2`

### Local LoRA Judge Inference

Use `src/inference/run_eval.py` when you want to evaluate a trained LoRA judge directly:

```bash
python src/inference/run_eval.py \
  --lora-model-path /path/to/lora_adapter \
  --base-model-path /path/to/base_vlm \
  --bmk editscore_consistency \
  --bmk-config $(pwd)/configs/datasets/bmk.json \
  --prompt-id vlm/assessment/visual_consistency/pairwise \
  --prompt-version v1 \
  --use-vllm
```

The repository also includes helper scripts such as `scripts/local_eval.sh` and `vllm_deploy_scripts/*.sh` for local serving and judge evaluation.

## Reproducibility

GEditBench v2 is structured to make experiment reruns and environment export manageable:

- environment profiles and lock files live under `environments/`
- pipeline and training behavior are configuration-driven
- prompt templates are versioned under `src/prompts/assets/`
- logs are collected under `logs/`
- cached intermediate results are stored in per-stage `.cache` directories
- historical training configs are archived under `configs/lora_sft/history/`

For a public release, we recommend pinning:

- model versions
- benchmark snapshots
- prompt versions
- training configs
- evaluation configs
- environment profile versions

## Roadmap

This repository is already usable as a research framework, but several public-release tasks are still worth completing:

- replace remaining internal absolute paths with environment-variable or config-template defaults
- sanitize and externalize private endpoints, credentials, and machine-specific settings
- provide public data-preparation instructions for benchmark assets not stored in this repository
- publish the accompanying paper and add a citation block
- add a license file and contribution guidelines for the first public release

## Citation

If you use GEditBench v2 in your research, please cite the accompanying paper once it is public. A BibTeX entry will be added here when the preprint is released.

## License

This repository does not yet include a published license file. Add a `LICENSE` file before the first public open-source release.
