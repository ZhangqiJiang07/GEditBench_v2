# Output Formats

In AutoPipeline, the most important interface is not a Python class but a file artifact. Downstream training, analysis, and manual inspection all depend on these outputs.

## 1. Annotation output

Path:

```text
<save-path>/<task>_grouped.jsonl
```

Simplified example:

```json
{
  "key": "sample_prompt_0001",
  "results": [
    {
      "source_image_path": "images/source/0001.png",
      "edited_image_path": "images/edited/model_a/0001.png",
      "instruction": "Add a red balloon to the left side of the boy.",
      "unedit_area": {
        "lpips": 0.084,
        "emd": 0.931,
        "ssim": 0.917
      }
    }
  ]
}
```

Characteristics:

- Each line represents one prompt or source-image group
- `results` is a list of candidate outcomes
- Human-centric outputs often include both `edit_area` and `unedit_area`

## 2. Eval output

Path:

```text
<save-path>/<bmk>/<config-name>/<timestamp>.jsonl
```

Simplified example:

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
    "gt_winner": "Image A",
    "winner": "Image B",
    "raw_responses": {
      "pair-judge": {
        "value": "Image B"
      }
    }
  }
}
```

Characteristics:

- Each line corresponds to one compared pair
- `winner` is the normalized final decision
- `raw_responses` keeps the original judge payload for debugging and auditing

## 3. Train-pairs output

Path:

```text
<output-dir>/<task>.json
```

Simplified example:

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

Characteristics:

- One JSON file is written per task
- The file can be consumed directly by preference-training workflows
- `gpt_response` preserves a standard winner representation

## 4. Cache files

In addition to the main outputs, the runtime writes cache files:

```text
<save-path>/.cache/*.jsonl
```

These are not intended as long-term public interfaces. Their role is to:

- resume interrupted runs
- skip already processed items
- avoid repeated calls to expensive modules or remote backends
