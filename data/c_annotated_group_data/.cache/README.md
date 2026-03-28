# Annotation Cache

This folder stores resumable cache files created by `autopipeline annotation`.

Typical file names

- `<task>_<pipeline>_results_cache.jsonl`

Cache entries are internal pipeline artifacts keyed by item id. They are derived data and can be regenerated from the corresponding candidate pool inputs if needed.
