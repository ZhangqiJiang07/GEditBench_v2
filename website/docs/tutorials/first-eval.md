# First Eval

`eval` is not designed to produce a large metric vector for a single image. Its job is to turn benchmark candidates into pairwise comparisons.

## When to use `eval`

Use `eval` when you need to answer questions such as:

- How does a model perform across a benchmark?
- Which of two candidate outputs is better?
- Can I obtain normalized winners directly for preference supervision?

## Required inputs

You need three categories of inputs:

1. A benchmark key, for example `openedit` or `vc_reward`
2. A `vlm-as-a-judge` pipeline config
3. A valid `user_config.yaml`

The benchmark registry is loaded from:

```text
<PROJECT_ROOT>/configs/datasets/bmk.json
```

Visible benchmark keys in the repository include:

- `openedit`
- `vc_reward`
- `editscore_consistency`
- `editscore_prompt_following`
- `editreward_visual_quality`

## Minimal command

```bash
cd <PROJECT_ROOT>
autopipeline eval \
  --bmk openedit \
  --pipeline-config-path <PROJECT_ROOT>/configs/pipelines/vlm_as_a_judge/openai.yaml \
  --user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml \
  --save-path <PROJECT_ROOT>/data/reward_eval_results
```

## Output file

```text
<save-path>/<bmk>/<config-name>/<timestamp>.jsonl
```

If you run `openedit`, the metadata filename is folded into the result directory name so that different metadata variants do not overwrite each other.

## Simplified output example

```json
{
  "key": "sample_prompt_0001_pair_model_a_vs_model_b",
  "results": {
    "input_dict": {
      "instruction": "Replace the mug with a glass cup.",
      "input_image": "images/source/0001.png",
      "edited_images": [
        "images/edited/model_a/0001.png",
        "images/edited/model_b/0001.png"
      ]
    },
    "winner": "Image B",
    "raw_responses": {
      "pair-judge": {
        "value": "Image B"
      }
    }
  }
}
```

## Common failure modes

- Benchmark paths exist, but the image assets are incomplete
- The `prompt_info` entry points to a prompt asset that does not exist
- The model backend is reachable, but the response does not satisfy the expected schema
- Too many predictions collapse to `Failed`

## Next steps

If your final goal is preference-data construction rather than benchmark reporting, the usual next step is [First Train Pairs](/docs/tutorials/first-train-pairs).
