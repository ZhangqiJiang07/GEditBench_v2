#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install.sh <annotate|train|pvc_judge> [--conda|--venv] [--with-optional] [--python <python>] [--venv-dir <path>]

Description:
  Thin wrapper around static environment manifests:
    conda: environments/<profile>.yml
    venv : environments/requirements/<profile>.lock.txt

Recommended direct commands:
  conda env create -f environments/annotate.yml
  python3.11 -m venv .venvs/annotate
  .venvs/annotate/bin/python -m pip install -r environments/requirements/annotate.lock.txt

Options:
  --conda                 Create or update the conda env from environments/<profile>.yml
  --venv                  Create or update a local venv and install from requirements/<profile>.lock.txt (default)
  --with-optional         Install environments/requirements/optional/<profile>.txt after the base env
  --python <python>       Python interpreter for --venv mode
  --venv-dir <path>       Override venv path (default: ./.venvs/<profile>)
  -h, --help              Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${PROJECT_ROOT}/environments"
REQ_ROOT="${ENV_ROOT}/requirements"
CONDA_BIN="${CONDA_BIN:-/data/miniforge3/bin/conda}"

MODE="venv"
WITH_OPTIONAL="0"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR=""

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
    --conda)
      MODE="conda"
      shift
      ;;
    --venv)
      MODE="venv"
      shift
      ;;
    --with-optional)
      WITH_OPTIONAL="1"
      shift
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="${2:-}"
      shift 2
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
    exit 1
    ;;
esac

profile_python_version() {
  case "$1" in
    annotate) echo "3.11" ;;
    train|pvc_judge) echo "3.12" ;;
  esac
}

profile_default_python_bin() {
  echo "python$(profile_python_version "$1")"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[install] Missing file: ${path}" >&2
    exit 1
  fi
}

has_requirements() {
  local path="$1"
  [[ -f "${path}" ]] && grep -Eq '^[[:space:]]*[^#[:space:]]' "${path}"
}

ensure_conda_bin() {
  if [[ -x "${CONDA_BIN}" ]]; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
    return 0
  fi
  echo "[install] conda not found. Install conda first or use --venv." >&2
  exit 1
}

ensure_python_bin() {
  if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(profile_default_python_bin "${PROFILE}")"
  fi

  if [[ -x "${PYTHON_BIN}" ]]; then
    return 0
  fi
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
    return 0
  fi

  echo "[install] Could not find interpreter: ${PYTHON_BIN}" >&2
  echo "[install] Install Python $(profile_python_version "${PROFILE}") or pass --python <path>." >&2
  exit 1
}

conda_env_exists() {
  local env_name="$1"
  "${CONDA_BIN}" env list | awk 'NR>2 {print $1}' | grep -Fxq "${env_name}"
}

install_optional_in_conda() {
  local req_file="$1"
  if [[ "${WITH_OPTIONAL}" != "1" ]] || ! has_requirements "${req_file}"; then
    return 0
  fi

  echo "[install] Installing optional dependencies: ${req_file}"
  "${CONDA_BIN}" run -n "${PROFILE}" python -m pip install -r "${req_file}"
}

install_optional_in_venv() {
  local python_exec="$1"
  local req_file="$2"
  if [[ "${WITH_OPTIONAL}" != "1" ]] || ! has_requirements "${req_file}"; then
    return 0
  fi

  echo "[install] Installing optional dependencies: ${req_file}"
  "${python_exec}" -m pip install -r "${req_file}"
}

install_extra_requirements_in_venv() {
  local python_exec="$1"
  local req_file="$2"
  if ! has_requirements "${req_file}"; then
    return 0
  fi

  echo "[install] Installing extra dependencies: ${req_file}"
  "${python_exec}" -m pip install -r "${req_file}"
}

ENV_FILE="${ENV_ROOT}/${PROFILE}.yml"
REQ_FILE="${REQ_ROOT}/${PROFILE}.lock.txt"
OPTIONAL_REQ_FILE="${REQ_ROOT}/optional/${PROFILE}.txt"
NON_PYPI_REQ_FILE="${REQ_ROOT}/non_pypi/${PROFILE}.txt"

require_file "${ENV_FILE}"
require_file "${REQ_FILE}"

if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${PROJECT_ROOT}/.venvs/${PROFILE}"
fi

if [[ "${MODE}" == "conda" ]]; then
  ensure_conda_bin
  if conda_env_exists "${PROFILE}"; then
    "${CONDA_BIN}" env update -n "${PROFILE}" -f "${ENV_FILE}" --prune
  else
    "${CONDA_BIN}" env create -f "${ENV_FILE}"
  fi
  install_optional_in_conda "${OPTIONAL_REQ_FILE}"

  conda_root="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
  echo "[install] Completed conda install for profile: ${PROFILE}"
  echo "[install] Activate with: source ${conda_root}/etc/profile.d/conda.sh && conda activate ${PROFILE}"
else
  ensure_python_bin
  mkdir -p "$(dirname "${VENV_DIR}")"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_DIR}/bin/python" -m pip install -r "${REQ_FILE}"
  install_extra_requirements_in_venv "${VENV_DIR}/bin/python" "${NON_PYPI_REQ_FILE}"
  install_optional_in_venv "${VENV_DIR}/bin/python" "${OPTIONAL_REQ_FILE}"

  echo "[install] Completed venv install for profile: ${PROFILE}"
  echo "[install] Activate with: source ${VENV_DIR}/bin/activate"
fi

if [[ "${WITH_OPTIONAL}" != "1" ]] && has_requirements "${OPTIONAL_REQ_FILE}"; then
  echo "[install] Optional dependencies are available at: ${OPTIONAL_REQ_FILE}"
fi
