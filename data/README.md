# Data Layout

This repository keeps the open-source data skeleton under `data/`, following the staged layout used in `/data/open_edit/data`.

Stage overview

- `a_raw_img_prompt_pair_data/`: raw image-instruction pools before filtering.
- `b_filtered_img_prompt_pair_data/`: filtered source pairs plus generated candidate metadata.
- `c_annotated_group_data/`: grouped annotation outputs and annotation caches.
- `d_train_data/`: pairwise training data consumed by `autotrain`.
- `e_openedit_pair_res/`: pairwise evaluation outputs on the GEditBench v2/OpenEdit benchmark.
- `f_reward_results/`: evaluation outputs on reward-model benchmarks such as `vc_reward`.
- `z_reward_bench/`: benchmark assets and shuffled benchmark annotations used by reward-benchmark loaders.

Path conventions

- Local paths, absolute paths, and remote URIs such as `s3://...` are all accepted by the codebase.
- JSONL files use one JSON object per line.
- Task names follow the checked-in config paths. In particular, the subject-editing task folders are `subject-add`, `subject-remove`, and `subject-replace`.

Typical file patterns

- `a_raw_img_prompt_pair_data/<task>.jsonl`
- `b_filtered_img_prompt_pair_data/<task>/meta_info.jsonl`
- `b_filtered_img_prompt_pair_data/<task>/<model>_generation_results.jsonl`
- `c_annotated_group_data/<task>_grouped.jsonl`
- `d_train_data/<task>.json`
- `e_openedit_pair_res/openedit/<eval_name>/<timestamp>.jsonl`
- `f_reward_results/<benchmark>/<eval_name>/<timestamp>.jsonl`

No benchmark or training data is shipped in this repository snapshot. These `README.md` files exist so the directory structure is preserved in Git and so downstream users know what to place where.
