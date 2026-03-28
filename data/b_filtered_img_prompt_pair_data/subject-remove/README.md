# subject-remove

This folder stores the filtered source set and generated candidates for the `subject-remove` task.

Typical files

- `meta_info.jsonl`
- `<model>_generation_results.jsonl`
- `gen_cache/<model>.jsonl`

Common JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/image.webp","instruction":"Remove the parked bicycle and fill the background naturally."}
```

This folder intentionally uses a hyphenated name because the checked-in candidate-pool configs reference `subject-remove/`. `meta_info.jsonl` references source images, while model result files reference edited outputs.
