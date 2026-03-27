# VLM-as-a-Judge Pipeline

`vlm-as-a-judge` is a pairwise pipeline. It does not produce a metric bundle for a single edited image. Instead, it answers a more direct question:

Which of two candidate edited images is better?

## Typical use cases

- comparing model outputs on a benchmark
- reward evaluation
- constructing pairwise preference data
- letting a VLM or LLM return a winner directly instead of hand-authoring complex score logic

## Starter configs

- `openai.yaml`
- `gemini.yaml`
- `vllm.yaml`
- `eval_if.yaml`
- `eval_vc.yaml`
- `eval_vq.yaml`

## Input and output contract

The key input fields are:

- `instruction`
- `input_image`
- `edited_images`

`edited_images` should contain the candidate images to compare.

The key output fields are:

- `winner`
- `raw_responses`

The normalized `winner` values are:

- `Image A`
- `Image B`
- `Tie`
- `Failed`

## Minimal example

```bash
cd <PROJECT_ROOT>
autopipeline eval \
  --bmk openedit \
  --pipeline-config-path <PROJECT_ROOT>/configs/pipelines/vlm_as_a_judge/openai.yaml \
  --user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml \
  --save-path <PROJECT_ROOT>/data/reward_eval_results
```

## How it differs from `annotation`

`annotation`:

- is better when you want structured metric scores
- supports metric-driven ranking and filtering

`vlm-as-a-judge`:

- is better when you want a direct pairwise winner
- is closer to preference modeling or benchmark ranking

## When not to use it

If your real requirement is to preserve fine-grained metric scores for each scope rather than collapse the decision into a final winner, start with `human-centric` or `object-centric` instead.
