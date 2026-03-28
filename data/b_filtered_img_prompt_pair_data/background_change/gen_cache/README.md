# background_change generation cache

This folder stores resumable generation cache files for the `background_change` task.

Typical file names

- `qwen-image-edit.jsonl`
- `kontext.jsonl`
- `bagel.jsonl`

Each line is a JSON object keyed by sample id and usually includes `key`, `image_path`, and `instruction`. These files are derived caches and can be regenerated.
