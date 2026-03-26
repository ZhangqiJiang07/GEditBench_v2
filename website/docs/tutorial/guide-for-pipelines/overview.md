# How to Choose a Pipeline

If you remember only one rule, remember this:

- Use `human-centric` or `object-centric` when you want structured scores for a single edited image
- Use `vlm-as-a-judge` when you want to compare two candidate edits directly

## Quick selection matrix

| Question | Recommended pipeline | Output |
| --- | --- | --- |
| Did a single edit satisfy an object-level instruction? | `object-centric` | scope -> metric -> score |
| Did a single human edit preserve identity, appearance, or local consistency? | `human-centric` | scope -> metric -> score |
| Which of two candidate edits is better? | `vlm-as-a-judge` | `winner` + `raw_responses` |

## When to choose `object-centric`

Typical tasks include:

- `subject_add`
- `subject_remove`
- `subject_replace`
- `color_alter`
- `material_alter`
- `size_adjustment`
- `text_editing`
- `cref` / `oref`

Key characteristics:

- Scores a single edited image at a time
- Relies primarily on parser-grounder plus general-purpose visual metric pipes
- Starter configs live in `configs/pipelines/object_centric/`

## When to choose `human-centric`

Choose this family for:

- `ps_human`
- `motion_change`
- other tasks that depend on human-local consistency

Key characteristics:

- Loads `face-detector`, `human-segmenter`, and `hair-segmenter`
- Separates `edit_area` from `unedit_area`
- Uses attribute-aware measurement rubrics to decide which metrics actually run

## When to choose `vlm-as-a-judge`

Choose this family for:

- benchmark pairwise evaluation
- direct A/B comparison of candidate outputs
- direct construction of pairwise training pairs

Key characteristics:

- Expects `instruction + input_image + edited_images`
- Returns `Image A`, `Image B`, `Tie`, or `Failed`
- Starter configs live in `configs/pipelines/vlm_as_a_judge/`

## A common confusion

Do not confuse "I want train pairs" with "which pipeline should I choose."

- `train-pairs` is a post-processing command
- The earlier choice between `annotation` and `eval` depends on whether you want structured scores or pairwise winners first

Continue with:

- [Object-Centric](/docs/tutorial/guide-for-pipelines/object-centric)
- [Human-Centric](/docs/tutorial/guide-for-pipelines/human-centric)
- [VLM-as-a-Judge](/docs/tutorial/guide-for-pipelines/vlm-as-a-judge)
