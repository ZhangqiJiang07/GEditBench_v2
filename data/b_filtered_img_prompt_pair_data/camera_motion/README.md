# camera_motion

This folder stores the filtered source set and generated candidates for the `camera_motion` task.

Typical files

- `meta_info.jsonl`
- `<model>_generation_results.jsonl`
- `gen_cache/<model>.jsonl`

Common JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/image.webp","instruction":"Simulate a leftward camera pan while keeping the scene coherent."}
```

`meta_info.jsonl` points to source images, while model result files point to edited outputs.
