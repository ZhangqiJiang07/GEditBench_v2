#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/export_envs.sh [annotate|train|pvc_judge|all]

Description:
  Export dependency files from source conda environments:
    annotate  <- pipeline3.11
    train     <- ft
    pvc_judge <- editscore

Outputs:
  environments/requirements/<profile>.lock.txt
  environments/requirements/optional/<profile>.txt
  environments/requirements/non_pypi/<profile>.txt
  environments/conda/<profile>.from-history.yml
  environments/conda/<profile>.explicit.txt
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${PROJECT_ROOT}/environments"
REQ_ROOT="${ENV_ROOT}/requirements"
CONDA_ROOT="${ENV_ROOT}/conda"
OPTIONAL_ROOT="${REQ_ROOT}/optional"
NON_PYPI_ROOT="${REQ_ROOT}/non_pypi"

CONDA_BIN="${CONDA_BIN:-/data/miniforge3/bin/conda}"
PUBLIC_PYPI_INDEX="${PUBLIC_PYPI_INDEX:-https://pypi.org/simple}"
PYTORCH_WHL_BASE="${PYTORCH_WHL_BASE:-https://download.pytorch.org/whl}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET="${1:-all}"

case "${TARGET}" in
  annotate|train|pvc_judge|all) ;;
  *)
    echo "[export] Unsupported target: ${TARGET}" >&2
    usage
    exit 1
    ;;
esac

if [[ ! -x "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  else
    echo "[export] conda not found. Set CONDA_BIN or install conda first." >&2
    exit 1
  fi
fi

mkdir -p "${REQ_ROOT}" "${OPTIONAL_ROOT}" "${CONDA_ROOT}" "${NON_PYPI_ROOT}"

profiles=(annotate train pvc_judge)
if [[ "${TARGET}" != "all" ]]; then
  profiles=("${TARGET}")
fi

source_env_for_profile() {
  case "$1" in
    annotate) echo "pipeline3.11" ;;
    train) echo "ft" ;;
    pvc_judge) echo "editscore" ;;
    *)
      echo "[export] Unknown profile: $1" >&2
      exit 1
      ;;
  esac
}

profile_python_version() {
  case "$1" in
    annotate) echo "3.11" ;;
    train|pvc_judge) echo "3.12" ;;
  esac
}

profile_torch_backend() {
  case "$1" in
    annotate) echo "cu126" ;;
    train|pvc_judge) echo "cu128" ;;
  esac
}

normalize_pkg_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-'
}

drop_from_base_requirements() {
  local profile="$1"
  local pkg_name="$2"

  case "${pkg_name}" in
    pip|setuptools|wheel|uv)
      return 0
      ;;
  esac

  if [[ "${profile}" == "annotate" && "${pkg_name}" == "sam-2" ]]; then
    return 0
  fi

  if [[ "${profile}" == "train" && "${pkg_name}" == "flash-attn" ]]; then
    return 0
  fi

  return 1
}

override_requirement_line() {
  local profile="$1"
  local pkg_name="$2"
  local line="$3"

  case "${profile}:${pkg_name}" in
    annotate:fsspec)
      echo "fsspec==2025.9.0"
      ;;
    annotate:numpy)
      echo "numpy==2.2.6"
      ;;
    train:av)
      echo "av==16.0.1"
      ;;
    train:torch)
      echo "torch==2.8.0+cu128"
      ;;
    train:torchaudio)
      echo "torchaudio==2.8.0+cu128"
      ;;
    train:torchvision)
      echo "torchvision==0.23.0+cu128"
      ;;
    *)
      echo "${line}"
      ;;
  esac
}

write_optional_requirements() {
  local profile="$1"
  local optional_file="${OPTIONAL_ROOT}/${profile}.txt"

  case "${profile}" in
    train)
      cat > "${optional_file}" <<'EOF'
# Optional accelerator packages for the train profile.
# Install them after the base environment with:
#   ./scripts/install.sh train --with-optional
flash_attn==2.8.3
EOF
      ;;
    *)
      cat > "${optional_file}" <<'EOF'
# No optional accelerator packages are currently defined for this profile.
EOF
      ;;
  esac
}

write_non_pypi_requirements() {
  local profile="$1"
  local non_pypi_file="${NON_PYPI_ROOT}/${profile}.txt"

  case "${profile}" in
    annotate)
      cat > "${non_pypi_file}" <<'EOF'
# No external non-PyPI dependencies are required.
# SAM2 is vendored in-repo under src/sam2.
EOF
      ;;
    *)
      cat > "${non_pypi_file}" <<'EOF'
# No external non-PyPI dependencies are required.
EOF
      ;;
  esac
}

normalize_lock_for_public_release() {
  local profile="$1"
  local lock_file="${REQ_ROOT}/${profile}.lock.txt"
  local tmp_file
  tmp_file="$(mktemp)"

  {
    echo "# Public install manifest for the ${profile} profile."
    echo "# Python $(profile_python_version "${profile}")"
    echo "# Generated from conda env $(source_env_for_profile "${profile}")"
    echo "--index-url ${PUBLIC_PYPI_INDEX}"
    echo "--extra-index-url ${PYTORCH_WHL_BASE}/$(profile_torch_backend "${profile}")"

    while IFS= read -r line || [[ -n "${line}" ]]; do
      [[ -z "${line}" ]] && continue
      local pkg_name raw_pkg
      raw_pkg="${line%%==*}"
      pkg_name="$(normalize_pkg_name "${raw_pkg}")"

      if drop_from_base_requirements "${profile}" "${pkg_name}"; then
        continue
      fi

      echo "$(override_requirement_line "${profile}" "${pkg_name}" "${line}")"
    done < "${lock_file}"
  } > "${tmp_file}"

  mv "${tmp_file}" "${lock_file}"
}

normalize_conda_history() {
  local profile="$1"
  local tmp_file
  tmp_file="$(mktemp)"

  "${CONDA_BIN}" env export -n "$(source_env_for_profile "${profile}")" --from-history > "${tmp_file}"
  awk -v env_name="${profile}" '
    /^name:/ { print "name: " env_name; next }
    /^prefix:/ { next }
    { print }
  ' "${tmp_file}" > "${CONDA_ROOT}/${profile}.from-history.yml"
  rm -f "${tmp_file}"

  if ! grep -Eq '^[[:space:]]*-[[:space:]]+pip([[:space:]]|=|$)' "${CONDA_ROOT}/${profile}.from-history.yml"; then
    printf '  - pip\n' >> "${CONDA_ROOT}/${profile}.from-history.yml"
  fi
}

for profile in "${profiles[@]}"; do
  source_env="$(source_env_for_profile "${profile}")"

  echo "[export] ${profile} <- ${source_env}"
  "${CONDA_BIN}" run -n "${source_env}" python -m pip list --format=freeze > "${REQ_ROOT}/${profile}.lock.txt"
  "${CONDA_BIN}" list -n "${source_env}" --explicit > "${CONDA_ROOT}/${profile}.explicit.txt"

  write_optional_requirements "${profile}"
  write_non_pypi_requirements "${profile}"
  normalize_lock_for_public_release "${profile}"
  normalize_conda_history "${profile}"
done

echo "[export] Done."
