# First Annotation

The goal of this page is straightforward: run `annotation` successfully once and understand why the output file looks the way it does.

## What `annotation` does

`annotation` is the right entry point when you need to:

- score single edited images with structured metrics
- filter or rank candidates based on those metrics
- prepare grouped results for downstream post-processing such as `train-pairs`

At the end of the run, it groups multiple candidate edits under the same source image and prompt into one `grouped.jsonl` file.

## Required inputs

You need at least three inputs:

1. `--edit-task`
2. A pipeline config that supports that task
3. A valid `user_config.yaml`

In most cases, you do not need to pass the candidate pool filename explicitly. The CLI automatically resolves `configs/datasets/candidate_pools/<edit-task>.json`.

## Minimal command

```bash
cd <PROJECT_ROOT>
autopipeline annotation \
  --edit-task subject_add \
  --pipeline-config-path <PROJECT_ROOT>/configs/pipelines/object_centric/subject_add.yaml \
  --user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml \
  --save-path <PROJECT_ROOT>/data/c_annotated_group_data
```

## What happens at runtime

You can think of the runtime in four steps:

1. The CLI normalizes `edit_task`
2. The runtime builds a dataset from the corresponding candidate pool
3. `ConfigEngine` resolves the pipeline config and user overrides
4. The runner executes the pipeline sample by sample and writes both cache files and grouped output

## Output files

Primary output:

```text
<save-path>/<task>_grouped.jsonl
```

Cache:

```text
<save-path>/.cache/<task>_<pipeline>_results_cache.jsonl
```

## Simplified output example

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

## Why some fields may be `null`

The most common reasons are:

- `parser-grounder` failed to localize the relevant region consistently
- A local expert did not obtain a valid bounding box or mask
- A metric was skipped by the measurement rubric for this sample

## Next steps

- If you need schema details, read [Output Formats](/docs/reference/output-formats)
- If you want to continue into preference-data construction, read [First Train Pairs](/docs/tutorials/first-train-pairs)
