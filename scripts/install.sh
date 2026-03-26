#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install.sh <annotate|train|pvc_judge> [options]

Options:
  --manager <auto|uv|pip|conda>      Install backend (default: auto)
  --conda-mode <history|explicit>     Conda input file type (default: history)
  --env-name <name>                   Conda environment name (default: profile)
  --venv-dir <path>                   venv path for uv/pip (default: ./.venvs/<profile>)
  --python-bin <python>               Python binary for pip mode (default: python3)
  --skip-non-pypi                     Skip Git/URL dependency install phase
  -h, --help                          Show this help

Examples:
  ./scripts/install.sh annotate
  ./scripts/install.sh train --manager pip
  ./scripts/install.sh pvc_judge --manager conda --env-name pvc_judge
  ./scripts/install.sh train --manager conda --conda-mode explicit
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${PROJECT_ROOT}/environments"
REQ_ROOT="${ENV_ROOT}/requirements"
CONDA_ROOT="${ENV_ROOT}/conda"

MANAGER="auto"
CONDA_MODE="history"
CONDA_BIN="${CONDA_BIN:-/data/miniforge3/bin/conda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_NAME=""
VENV_DIR=""
SKIP_NON_PYPI="0"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PROFILE="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manager)
      MANAGER="${2:-}"
      shift 2
      ;;
    --conda-mode)
      CONDA_MODE="${2:-}"
      shift 2
      ;;
    --env-name)
      ENV_NAME="${2:-}"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --skip-non-pypi)
      SKIP_NON_PYPI="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[install] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${PROFILE}" in
  annotate|train|pvc_judge) ;;
  *)
    echo "[install] Unsupported profile: ${PROFILE}" >&2
    echo "[install] Supported profiles: annotate, train, pvc_judge" >&2
    exit 1
    ;;
esac

profile_to_uv_extra() {
  case "$1" in
    annotate) echo "annotate" ;;
    train) echo "train" ;;
    pvc_judge) echo "pvc-judge" ;;
    *)
      echo "[install] Unsupported profile for uv extra: $1" >&2
      exit 1
      ;;
  esac
}

case "${MANAGER}" in
  auto|uv|pip|conda) ;;
  *)
    echo "[install] Unsupported manager: ${MANAGER}" >&2
    exit 1
    ;;
esac

case "${CONDA_MODE}" in
  history|explicit) ;;
  *)
    echo "[install] Unsupported conda mode: ${CONDA_MODE}" >&2
    exit 1
    ;;
esac

if [[ -z "${ENV_NAME}" ]]; then
  ENV_NAME="${PROFILE}"
fi

if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${PROJECT_ROOT}/.venvs/${PROFILE}"
fi

REQ_FILE="${REQ_ROOT}/${PROFILE}.lock.txt"
CONDA_HISTORY_FILE="${CONDA_ROOT}/${PROFILE}.from-history.yml"
CONDA_EXPLICIT_FILE="${CONDA_ROOT}/${PROFILE}.explicit.txt"
NON_PYPI_FILE="${REQ_ROOT}/non_pypi/${PROFILE}.txt"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[install] Missing file: ${path}" >&2
    exit 1
  fi
}

ensure_conda_bin() {
  if [[ -x "${CONDA_BIN}" ]]; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
    return 0
  fi
  echo "[install] conda not found. Set CONDA_BIN or install conda first." >&2
  exit 1
}

conda_env_exists() {
  local env_name="$1"
  "${CONDA_BIN}" env list | awk 'NR>2 {print $1}' | grep -Fxq "${env_name}"
}

install_with_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "[install] uv not found. Use --manager pip or install uv first." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${VENV_DIR}")"

  if [[ -f "${PROJECT_ROOT}/pyproject.toml" && -f "${PROJECT_ROOT}/uv.lock" ]]; then
    uv_extra="$(profile_to_uv_extra "${PROFILE}")"
    uv venv "${VENV_DIR}"
    uv sync --frozen --extra "${uv_extra}" --python "${VENV_DIR}/bin/python"
  else
    require_file "${REQ_FILE}"
    uv venv "${VENV_DIR}"
    uv pip install --python "${VENV_DIR}/bin/python" -r "${REQ_FILE}"
  fi

  echo "[install] Completed with uv for profile: ${PROFILE}"
  echo "[install] Activate with: source ${VENV_DIR}/bin/activate"
}

install_with_pip() {
  require_file "${REQ_FILE}"
  mkdir -p "$(dirname "${VENV_DIR}")"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_DIR}/bin/pip" install -r "${REQ_FILE}"
  echo "[install] Completed with pip for profile: ${PROFILE}"
  echo "[install] Activate with: source ${VENV_DIR}/bin/activate"
}

install_with_conda_history() {
  require_file "${CONDA_HISTORY_FILE}"
  local tmp_file
  tmp_file="$(mktemp)"
  awk -v env_name="${ENV_NAME}" '
    /^name:/ { print "name: " env_name; next }
    /^prefix:/ { next }
    { print }
  ' "${CONDA_HISTORY_FILE}" > "${tmp_file}"

  if conda_env_exists "${ENV_NAME}"; then
    "${CONDA_BIN}" env update --name "${ENV_NAME}" --file "${tmp_file}" --prune
  else
    "${CONDA_BIN}" env create --name "${ENV_NAME}" --file "${tmp_file}"
  fi
  rm -f "${tmp_file}"
}

install_with_conda_explicit() {
  require_file "${CONDA_EXPLICIT_FILE}"
  if conda_env_exists "${ENV_NAME}"; then
    "${CONDA_BIN}" install --name "${ENV_NAME}" --file "${CONDA_EXPLICIT_FILE}" -y
  else
    "${CONDA_BIN}" create --name "${ENV_NAME}" --file "${CONDA_EXPLICIT_FILE}" -y
  fi
}

install_with_conda() {
  ensure_conda_bin
  if [[ "${CONDA_MODE}" == "history" ]]; then
    install_with_conda_history
  else
    install_with_conda_explicit
  fi
  echo "[install] Completed with conda for profile: ${PROFILE}"
  echo "[install] Activate with: source /data/miniforge3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
}

install_non_pypi_deps() {
  if [[ "${SKIP_NON_PYPI}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${NON_PYPI_FILE}" ]]; then
    return 0
  fi
  if ! grep -Eq '^[[:space:]]*[^#[:space:]]' "${NON_PYPI_FILE}"; then
    return 0
  fi

  echo "[install] Installing non-PyPI dependencies: ${NON_PYPI_FILE}"
  case "${MANAGER}" in
    uv)
      uv pip install --python "${VENV_DIR}/bin/python" -r "${NON_PYPI_FILE}"
      ;;
    pip)
      "${VENV_DIR}/bin/pip" install -r "${NON_PYPI_FILE}"
      ;;
    conda)
      ensure_conda_bin
      "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r "${NON_PYPI_FILE}"
      ;;
  esac
}

if [[ "${MANAGER}" == "auto" ]]; then
  if command -v uv >/dev/null 2>&1; then
    MANAGER="uv"
  else
    MANAGER="pip"
  fi
fi

case "${MANAGER}" in
  uv)
    install_with_uv
    ;;
  pip)
    install_with_pip
    ;;
  conda)
    install_with_conda
    ;;
esac

install_non_pypi_deps
