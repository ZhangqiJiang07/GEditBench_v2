#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}:${SRC_ROOT}"
fi

FILE_PATH="$1"

python ${SRC_ROOT}/common_utils/calculate_statistics.py \
  --file-path "$FILE_PATH"