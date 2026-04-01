# Reward Benchmark Assets

This folder stores the benchmark assets and annotations used by reward-benchmark loaders.

Typical contents

- `editreward_bench_visual_quality.jsonl`: pairwise visual-quality benchmark annotations.
- `shuffled_editscore_bench_consistency.json`: shuffle map for EditScore consistency evaluation.
- `shuffled_editscore_bench_prompt_following.json`: shuffle map for EditScore prompt-following evaluation.

These names are referenced directly by `configs/datasets/bmk.json`.
