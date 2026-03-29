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

RESULTS_ROOT="${REPO_ROOT}/data/e_geditv2_pair_res/geditv2"
IF_RESULTS_PATH="${RESULTS_ROOT}/eval_if_metadata/example.jsonl"
VQ_RESULTS_PATH="${RESULTS_ROOT}/eval_vq_metadata/example.jsonl"
VC_RESULTS_PATH="${RESULTS_ROOT}/eval_vc_metadata/example.jsonl"

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
