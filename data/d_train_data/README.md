# Pairwise Training Data

This folder stores the JSON training sets produced by `autopipeline train-pairs` and consumed by `autotrain`.

Typical file names

- `<task>.json`
- `<task>_1500prompts.json`
- `validation_set.json`

Expected JSON schema

Each file is a JSON array. Each element represents a pairwise preference example:

```json
{
  "source_image_path":"path/or/uri/to/source_image.png",
  "edited_image_paths":["path/to/image_a.png","path/to/image_b.png"],
  "instruction":"Edit instruction text",
  "gpt_response":"{\"winner\": \"Image A\"}"
}
```

`edited_image_paths[0]` and `edited_image_paths[1]` are ordered to match the winner encoded in `gpt_response`.
