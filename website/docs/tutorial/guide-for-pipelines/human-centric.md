# Human-Centric Pipeline

`human-centric` is the pipeline family for person and body editing. Compared with `object-centric`, the difference is not just a different metric list. The entire runtime also depends on human-specific experts and cropped local regions.

## What it is designed to measure

Typical questions include:

- Was facial identity preserved?
- Did facial geometry or facial texture degrade?
- Is the edited hair region consistent?
- Did body pose and appearance remain coherent?
- Which unedited regions still need evaluation after a large human edit?

## Starter configs

- `configs/pipelines/human_centric/ps_human.yaml`
- `configs/pipelines/human_centric/motion_change.yaml`

## What makes this family different

Unlike `object-centric`, this family requires `expert_configs` for:

- `face-detector`
- `human-segmenter`
- `hair-segmenter`

It also separates runtime measurements into:

- `edit_area`
- `unedit_area`

The pipeline uses parsed edit attributes to assemble a measurement rubric, which determines which metrics should actually run for a given sample.

## Minimal example

```bash
cd <PROJECT_ROOT>
autopipeline annotation \
  --edit-task motion_change \
  --pipeline-config-path <PROJECT_ROOT>/configs/pipelines/human_centric/motion_change.yaml \
  --user-config <PROJECT_ROOT>/configs/pipelines/user_config.yaml \
  --save-path <PROJECT_ROOT>/data/c_annotated_group_data
```

## What you will see in the results

The output is still grouped JSONL, but the score structure usually includes both:

- `edit_area`
- `unedit_area`

Some metrics may still be `null`. That does not necessarily indicate a broken run. Common reasons include:

- face bounding boxes were not detected
- segmentation masks were unavailable
- the measurement rubric skipped a metric for that edit type

## When to prefer it

If your quality criteria depend on human-local details, use `human-centric` first instead of forcing the task into `object-centric`.

If your real question is which of two candidate images is better rather than how to score one image structurally, continue with [VLM-as-a-Judge](/docs/tutorial/guide-for-pipelines/vlm-as-a-judge).
