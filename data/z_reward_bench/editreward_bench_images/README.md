# EditReward-Bench Image Tree

This folder stores the actual images referenced by `data/z_reward_bench/editreward_bench_visual_quality.jsonl`.

Recommended layout

- `input/<key>.png`: source images
- `candidate_1/<key>.png`: first edited candidate
- `candidate_2/<key>.png`: second edited candidate

The JSONL file may also point to absolute paths or remote URIs, but this local directory is the canonical in-repo placeholder.
