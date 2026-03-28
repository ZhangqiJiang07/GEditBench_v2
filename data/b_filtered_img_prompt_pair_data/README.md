# Filtered Source Pairs And Candidate Metadata

This folder stores task-specific filtered source sets and the generated candidate metadata built on top of them.

Each task directory usually contains

- `meta_info.jsonl`: the filtered source set used as input to candidate generation.
- `<model>_generation_results.jsonl`: one JSONL file per generation backend.
- `gen_cache/`: incremental generation cache files written while a model run is in progress.

Task directories in this open-source layout

- `background_change`
- `camera_motion`
- `color_alter`
- `cref`
- `enhancement`
- `material_alter`
- `motion_change`
- `object_extraction`
- `oref`
- `ps_human`
- `size_adjustment`
- `sref`
- `style_transfer`
- `subject-add`
- `subject-remove`
- `subject-replace`
- `text_editing`
- `tone_transfer`

The folder names above follow the checked-in candidate-pool configs and should not be renamed without updating `configs/datasets/candidate_pools/*.json`.
