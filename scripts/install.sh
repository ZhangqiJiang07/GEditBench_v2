#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install.sh <annotate|train|pvc_judge> [options]

Options:
  --manager <auto|uv|pip|conda>      Install backend (default: auto)
  --conda-mode <history|explicit>    Conda input file type (default: history)
  --env-name <name>                  Conda environment name (default: profile)
  --venv-dir <path>                  venv path for uv/pip (default: ./.venvs/<profile>)
  --python-bin <python>              Python binary for pip mode
  --tmp-dir <path>                   Temporary directory for large package installs
  --skip-non-pypi                    Skip Git/URL dependency install phase
  --with-optional                    Install optional accelerator packages when available
  -h, --help                         Show this help

Examples:
  ./scripts/install.sh annotate
  ./scripts/install.sh train --manager conda --env-name train
  ./scripts/install.sh train --manager uv --with-optional
  ./scripts/install.sh pvc_judge --manager pip --python-bin python3.12
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
ENV_NAME=""
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR=""
INSTALL_TMPDIR="${OPEN_EDIT_TMPDIR:-}"
SKIP_NON_PYPI="0"
WITH_OPTIONAL="0"
PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

PUBLIC_PYPI_INDEX="${PUBLIC_PYPI_INDEX:-https://pypi.org/simple}"
PYTORCH_WHL_BASE="${PYTORCH_WHL_BASE:-https://download.pytorch.org/whl}"

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
    --tmp-dir)
      INSTALL_TMPDIR="${2:-}"
      shift 2
      ;;
    --skip-non-pypi)
      SKIP_NON_PYPI="1"
      shift
      ;;
    --with-optional)
      WITH_OPTIONAL="1"
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

profile_python_version() {
  case "$1" in
    annotate) echo "3.11" ;;
    train|pvc_judge) echo "3.12" ;;
  esac
}

profile_default_python_bin() {
  echo "python$(profile_python_version "$1")"
}

profile_torch_backend() {
  case "$1" in
    annotate) echo "cu126" ;;
    train|pvc_judge) echo "cu128" ;;
  esac
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

  echo "[install] Python ${PROFILE_PYTHON} is required for profile ${PROFILE}." >&2
  echo "[install] Could not find interpreter: ${PYTHON_BIN}" >&2
  echo "[install] Use --python-bin <path>, or use --manager uv/conda instead." >&2
  exit 1
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[install] Missing file: ${path}" >&2
    exit 1
  fi
}

setup_runtime_dirs() {
  local cache_root

  cache_root="${OPEN_EDIT_CACHE_DIR:-${PROJECT_ROOT}/.cache/open_edit}"
  mkdir -p "${cache_root}"

  if [[ -z "${INSTALL_TMPDIR}" ]]; then
    INSTALL_TMPDIR="${cache_root}/tmp"
  fi

  mkdir -p "${INSTALL_TMPDIR}"
  INSTALL_TMPDIR="$(cd "${INSTALL_TMPDIR}" && pwd)"

  export TMPDIR="${INSTALL_TMPDIR}"
  export TMP="${INSTALL_TMPDIR}"
  export TEMP="${INSTALL_TMPDIR}"

  if [[ -z "${UV_CACHE_DIR:-}" ]]; then
    export UV_CACHE_DIR="${cache_root}/uv"
    mkdir -p "${UV_CACHE_DIR}"
  fi
}

conda_env_exists() {
  local env_name="$1"
  "${CONDA_BIN}" env list | awk 'NR>2 {print $1}' | grep -Fxq "${env_name}"
}

run_public_pip() {
  local python_exec="$1"
  shift
  PIP_CONFIG_FILE=/dev/null \
    PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR}" \
    "${python_exec}" -m pip --isolated --no-cache-dir "$@"
}

run_public_pip_in_conda() {
  PIP_CONFIG_FILE=/dev/null \
    PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR}" \
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip --isolated --no-cache-dir "$@"
}

install_requirements_with_pip() {
  local python_exec="$1"
  local req_file="$2"

  require_file "${req_file}"
  run_public_pip "${python_exec}" install --upgrade pip setuptools wheel
  run_public_pip "${python_exec}" install --prefer-binary -r "${req_file}"
}

install_requirements_in_conda() {
  local req_file="$1"

  require_file "${req_file}"
  run_public_pip_in_conda install --upgrade pip setuptools wheel
  run_public_pip_in_conda install --prefer-binary -r "${req_file}"
}

install_requirements_with_uv() {
  local python_exec="$1"
  local req_file="$2"

  require_file "${req_file}"
  uv pip install \
    --no-config \
    --python "${python_exec}" \
    --default-index "${PUBLIC_PYPI_INDEX}" \
    --extra-index-url "${PROFILE_TORCH_INDEX}" \
    -r "${req_file}"
}

install_optional_with_pip() {
  local python_exec="$1"
  local req_file="$2"

  if [[ "${WITH_OPTIONAL}" != "1" || ! -f "${req_file}" ]]; then
    return 0
  fi
  if ! grep -Eq '^[[:space:]]*[^#[:space:]]' "${req_file}"; then
    return 0
  fi

  echo "[install] Installing optional dependencies: ${req_file}"
  run_public_pip "${python_exec}" install --prefer-binary --no-build-isolation -r "${req_file}"
}

install_optional_in_conda() {
  local req_file="$1"

  if [[ "${WITH_OPTIONAL}" != "1" || ! -f "${req_file}" ]]; then
    return 0
  fi
  if ! grep -Eq '^[[:space:]]*[^#[:space:]]' "${req_file}"; then
    return 0
  fi

  echo "[install] Installing optional dependencies: ${req_file}"
  run_public_pip_in_conda install --prefer-binary --no-build-isolation -r "${req_file}"
}

install_optional_with_uv() {
  local python_exec="$1"
  local req_file="$2"

  if [[ "${WITH_OPTIONAL}" != "1" || ! -f "${req_file}" ]]; then
    return 0
  fi
  if ! grep -Eq '^[[:space:]]*[^#[:space:]]' "${req_file}"; then
    return 0
  fi

  echo "[install] Installing optional dependencies: ${req_file}"
  uv pip install \
    --no-config \
    --python "${python_exec}" \
    --default-index "${PUBLIC_PYPI_INDEX}" \
    --extra-index-url "${PROFILE_TORCH_INDEX}" \
    --no-build-isolation \
    -r "${req_file}"
}

install_non_pypi_with_pip() {
  local python_exec="$1"

  if [[ "${SKIP_NON_PYPI}" == "1" || ! -f "${NON_PYPI_FILE}" ]]; then
    return 0
  fi
  if ! grep -Eq '^[[:space:]]*[^#[:space:]]' "${NON_PYPI_FILE}"; then
    return 0
  fi

  echo "[install] Installing non-PyPI dependencies: ${NON_PYPI_FILE}"
  run_public_pip "${python_exec}" install -r "${NON_PYPI_FILE}"
}

install_non_pypi_in_conda() {
  if [[ "${SKIP_NON_PYPI}" == "1" || ! -f "${NON_PYPI_FILE}" ]]; then
    return 0
  fi
  if ! grep -Eq '^[[:space:]]*[^#[:space:]]' "${NON_PYPI_FILE}"; then
    return 0
  fi

  echo "[install] Installing non-PyPI dependencies: ${NON_PYPI_FILE}"
  run_public_pip_in_conda install -r "${NON_PYPI_FILE}"
}

install_non_pypi_with_uv() {
  local python_exec="$1"

  if [[ "${SKIP_NON_PYPI}" == "1" || ! -f "${NON_PYPI_FILE}" ]]; then
    return 0
  fi
  if ! grep -Eq '^[[:space:]]*[^#[:space:]]' "${NON_PYPI_FILE}"; then
    return 0
  fi

  echo "[install] Installing non-PyPI dependencies: ${NON_PYPI_FILE}"
  uv pip install --no-config --python "${python_exec}" -r "${NON_PYPI_FILE}"
}

link_project_src_with_python() {
  local python_exec="$1"
  local site_packages

  site_packages="$(
    "${python_exec}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
  )"

  printf '%s\n' "${PROJECT_ROOT}/src" > "${site_packages}/open_edit_local_src.pth"
}

link_project_src_in_conda() {
  local site_packages

  site_packages="$(
    "${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
  )"

  printf '%s\n' "${PROJECT_ROOT}/src" > "${site_packages}/open_edit_local_src.pth"
}

install_with_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "[install] uv not found. Use --manager pip/conda or install uv first." >&2
    exit 1
  fi

  mkdir -p "$(dirname "${VENV_DIR}")"
  uv venv --no-config --python "${PROFILE_PYTHON}" "${VENV_DIR}"

  install_requirements_with_uv "${VENV_DIR}/bin/python" "${REQ_FILE}"
  install_optional_with_uv "${VENV_DIR}/bin/python" "${OPTIONAL_REQ_FILE}"
  install_non_pypi_with_uv "${VENV_DIR}/bin/python"
  link_project_src_with_python "${VENV_DIR}/bin/python"

  echo "[install] Completed with uv for profile: ${PROFILE}"
  echo "[install] Activate with: source ${VENV_DIR}/bin/activate"
}

install_with_pip() {
  ensure_python_bin
  mkdir -p "$(dirname "${VENV_DIR}")"

  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  install_requirements_with_pip "${VENV_DIR}/bin/python" "${REQ_FILE}"
  install_optional_with_pip "${VENV_DIR}/bin/python" "${OPTIONAL_REQ_FILE}"
  install_non_pypi_with_pip "${VENV_DIR}/bin/python"
  link_project_src_with_python "${VENV_DIR}/bin/python"

  echo "[install] Completed with pip for profile: ${PROFILE}"
  echo "[install] Activate with: source ${VENV_DIR}/bin/activate"
}

install_with_conda_history() {
  require_file "${CONDA_HISTORY_FILE}"

  local tmp_file
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/open_edit_conda_history.XXXXXX.yml")"
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

  install_requirements_in_conda "${REQ_FILE}"
  install_optional_in_conda "${OPTIONAL_REQ_FILE}"
  install_non_pypi_in_conda
  link_project_src_in_conda

  local conda_root
  conda_root="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
  echo "[install] Completed with conda for profile: ${PROFILE}"
  echo "[install] Activate with: source ${conda_root}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
}

PROFILE_PYTHON="$(profile_python_version "${PROFILE}")"
PROFILE_TORCH_INDEX="${PYTORCH_WHL_BASE}/$(profile_torch_backend "${PROFILE}")"

if [[ -z "${ENV_NAME}" ]]; then
  ENV_NAME="${PROFILE}"
fi

if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${PROJECT_ROOT}/.venvs/${PROFILE}"
fi

REQ_FILE="${REQ_ROOT}/${PROFILE}.lock.txt"
OPTIONAL_REQ_FILE="${REQ_ROOT}/optional/${PROFILE}.txt"
NON_PYPI_FILE="${REQ_ROOT}/non_pypi/${PROFILE}.txt"
CONDA_HISTORY_FILE="${CONDA_ROOT}/${PROFILE}.from-history.yml"
CONDA_EXPLICIT_FILE="${CONDA_ROOT}/${PROFILE}.explicit.txt"

if [[ "${MANAGER}" == "auto" ]]; then
  if command -v uv >/dev/null 2>&1; then
    MANAGER="uv"
  elif [[ -x "${CONDA_BIN}" ]] || command -v conda >/dev/null 2>&1; then
    MANAGER="conda"
  else
    MANAGER="pip"
  fi
fi

setup_runtime_dirs

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

if [[ "${WITH_OPTIONAL}" != "1" && -f "${OPTIONAL_REQ_FILE}" ]] && grep -Eq '^[[:space:]]*[^#[:space:]]' "${OPTIONAL_REQ_FILE}"; then
  echo "[install] Optional accelerators are available at: ${OPTIONAL_REQ_FILE}"
  echo "[install] Install them with: ./scripts/install.sh ${PROFILE} --manager ${MANAGER} --with-optional"
fi
