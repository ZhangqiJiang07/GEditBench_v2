# GEditBench v2 / OpenEdit Pairwise Evaluation Results

This folder stores pairwise evaluation outputs on the GEditBench v2 benchmark. The directory name follows the legacy `open_edit/data` layout even though the benchmark itself is named GEditBench v2 in this repository.

Typical contents

- `openedit/`: metric-specific evaluation result folders.
- `.cache/`: cached pairwise evaluation results for resumable runs.

Canonical output pattern

- `openedit/eval_if_metadata/<timestamp>.jsonl`
- `openedit/eval_vc_metadata/<timestamp>.jsonl`
- `openedit/eval_vq_metadata/<timestamp>.jsonl`
