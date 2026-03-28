# color_alter

This folder stores the filtered source set and generated candidates for the `color_alter` task.

Typical files

- `meta_info.jsonl`
- `<model>_generation_results.jsonl`
- `gen_cache/<model>.jsonl`

Common JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/image.webp","instruction":"Change the car to metallic cherry red while preserving the scene lighting."}
```

`meta_info.jsonl` references source images. Model result files reference edited outputs.
