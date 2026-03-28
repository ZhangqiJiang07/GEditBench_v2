# EditReward-Bench Dataset Export

This folder is reserved for a benchmark export created with Hugging Face Datasets `save_to_disk(...)`.

Expected contents

- Arrow or parquet shard files
- `dataset_info.json`
- `state.json`
- Split metadata used by `datasets.load_from_disk(...)`

The EditScore benchmark loader reads this directory directly rather than a single JSONL file.
