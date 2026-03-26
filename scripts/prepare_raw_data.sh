#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

usage() {
  cat <<'EOF'
Usage:
  prepare_raw_data.sh --dataset <pico|nano|unicedit> [dataset_args...]
  prepare_raw_data.sh <pico|nano|unicedit> [dataset_args...]

Examples:
  prepare_raw_data.sh --dataset nano --task background_change --path-to-nano-data /path/to/Nano --output-dir /path/to/save/metadata_files --image-save-path /path/to/save/the/downloaded/data/
  prepare_raw_data.sh pico --task subject-add --image-save-path ${REPO_ROOT}/data/images --output-dir /path/to/save/metadata_files --path-to-pico-sft-jsonl /path/to/pico/sft.jsonl
  prepare_raw_data.sh unicedit --path-to-uniedit-data /path/to/UnicEdit-10M/data --output-dir ${REPO_ROOT}/data/a_raw_img_prompt_pair_data/UnicEdit-10M --max-workers 16

Notes:
  - This script auto-sets PYTHONPATH to include repo root and src.
  - Use PYTHON_BIN to choose interpreter, default is `python3`.
EOF
}

DATASET=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "Error: --dataset requires a value." >&2
        exit 1
      fi
      DATASET="$2"
      shift 2
      ;;
    -h|--help)
      if [[ -z "${DATASET}" ]]; then
        usage
        exit 0
      fi
      break
      ;;
    --)
      shift
      break
      ;;
    *)
      if [[ -z "${DATASET}" ]]; then
        DATASET="$1"
        shift
      fi
      break
      ;;
  esac
done

if [[ -z "${DATASET}" ]]; then
  echo "Error: missing dataset. Use --dataset <pico|nano|unicedit>." >&2
  exit 1
fi

case "${DATASET}" in
  pico)
    TARGET_SCRIPT="${REPO_ROOT}/src/autogen/prepare_pico_data.py"
    ;;
  nano)
    TARGET_SCRIPT="${REPO_ROOT}/src/autogen/prepare_nano_consistent_data.py"
    ;;
  unicedit)
    TARGET_SCRIPT="${REPO_ROOT}/src/autogen/prepare_unicedit.py"
    ;;
  *)
    echo "Error: unsupported dataset '${DATASET}'. Choose from: pico, nano, unicedit." >&2
    exit 1
    ;;
esac

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}"
fi

exec "${PYTHON_BIN}" "${TARGET_SCRIPT}" "$@"
