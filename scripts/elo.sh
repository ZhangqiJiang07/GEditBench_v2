#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}"
fi

PYTHON_BIN=${PYTHON_BIN:-python3}

BOOTSTRAP=${1:-100}
EXCLUDE_MODELS=${2:-}

IF_RESULTS_PATH="/data/open_edit/data/e_openedit_pair_res/openedit_eval_if_metadata_20260321_172730.jsonl"
VQ_RESULTS_PATH="/data/open_edit/data/e_openedit_pair_res/openedit/eval_vq_metadata/20260323_202353.jsonl"
VC_RESULTS_PATH="/data/open_edit/data/e_openedit_pair_res/openedit/eval_vc_metadata/20260326_213931.jsonl"

CMD=(
  "${PYTHON_BIN}" "${SRC_ROOT}/common_utils/elo_score.py"
  --result-files "$VC_RESULTS_PATH,$VQ_RESULTS_PATH,$IF_RESULTS_PATH"
  --bootstrap "$BOOTSTRAP"
  --alpha 1
  --dimension-weighting "balanced"
  --seed 42
)

if [[ -n "${EXCLUDE_MODELS}" ]]; then
  CMD+=(--exclude-models "${EXCLUDE_MODELS}")
fi

"${CMD[@]}"
