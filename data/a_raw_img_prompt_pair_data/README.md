# Raw Image-Prompt Pairs

This folder stores the raw image-instruction pools before diversity filtering.

Typical contents

- `<task>.jsonl`: raw source pairs for one edit task.
- Optional dataset-specific exports such as `UnicEdit-10M/` if you materialize raw public datasets locally.

Expected JSONL schema

```json
{"key":"sample_0001","image_path":"path/or/uri/to/source_image.jpg","instruction":"Edit instruction text"}
```

Additional fields are allowed. For example, preparation scripts may also keep dataset-specific metadata such as `detailed_instruction`, `parquet_file`, or `row_idx`. The downstream filtering step only requires a stable `key`, a source image reference, and the instruction text.
