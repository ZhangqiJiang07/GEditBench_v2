# motion_change

This folder stores the filtered source set and generated candidates for the `motion_change` task.

Typical files

- `meta_info.jsonl`
- `<model>_generation_results.jsonl`
- `gen_cache/<model>.jsonl`

Common JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/image.webp","instruction":"Change the person's pose to a running motion while keeping identity consistent."}
```

`meta_info.jsonl` references source images. Model result files reference edited outputs.
