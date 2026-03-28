# sref

This folder stores the filtered source set and generated candidates for the `sref` task.

Typical files

- `meta_info.jsonl`
- `<model>_generation_results.jsonl`
- `gen_cache/<model>.jsonl`

Common JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/image.webp","instruction":"Edit the scene to match the provided style reference."}
```

`meta_info.jsonl` references source images. Model result files reference edited outputs.
