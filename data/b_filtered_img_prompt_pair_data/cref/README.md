# cref

This folder stores the filtered source set and generated candidates for the `cref` task.

Typical files

- `meta_info.jsonl`
- `<model>_generation_results.jsonl`
- `gen_cache/<model>.jsonl`

Common JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/image.webp","instruction":"Make the edited subject match the provided character reference."}
```

`meta_info.jsonl` references source images. Model result files reference edited outputs.
