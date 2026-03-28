# subject-replace

This folder stores the filtered source set and generated candidates for the `subject-replace` task.

Typical files

- `meta_info.jsonl`
- `<model>_generation_results.jsonl`
- `gen_cache/<model>.jsonl`

Common JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/image.webp","instruction":"Replace the cat with a corgi and keep the pose natural."}
```

This folder intentionally uses a hyphenated name because the checked-in candidate-pool configs reference `subject-replace/`. `meta_info.jsonl` references source images, while model result files reference edited outputs.
