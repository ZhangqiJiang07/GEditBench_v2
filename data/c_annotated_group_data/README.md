# Grouped Annotation Outputs

This folder stores the outputs produced by `autopipeline annotation`.

Typical contents

- `<task>_grouped.jsonl`: grouped annotation results for one task.
- `.cache/`: per-item pipeline cache files used for resumable annotation runs.

Grouped JSONL schema

- Object-centric and human-centric pipelines write one line per source sample:

```json
{"key":"sample_0001","results":[{"source_image_path":"...","edited_image_path":"...","instruction":"...","unedit_area":{"lpips":0.12},"edit_area":{"dinov3_structure_similarity":0.91}}]}
```

- Judge-style pipelines write one line per pairwise comparison:

```json
{"key":"sample_0001_pair_ModelA_vs_ModelB","results":{"input_dict":{"instruction":"...","input_image":"...","edited_images":["...","..."]},"winner":"Image A","raw_responses":{"pair-judge":{"type":"pairwise_comparison","value":"Image A"}}}}
```

The exact metric keys depend on the selected pipeline config.
