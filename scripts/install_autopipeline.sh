#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_DIR="${HOME}/.local/bin"
TARGET_PATH="${TARGET_DIR}/autopipeline"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NO_PATH_UPDATE="${NO_PATH_UPDATE:-0}"

mkdir -p "${TARGET_DIR}"

cat > "${TARGET_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT}"
cd "\${PROJECT_ROOT}"
if [[ -n "\${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="\${PROJECT_ROOT}:\${PROJECT_ROOT}/src:\${PYTHONPATH}"
else
  export PYTHONPATH="\${PROJECT_ROOT}:\${PROJECT_ROOT}/src"
fi
exec "${PYTHON_BIN}" -m src.cli.autopipeline "\$@"
EOF

chmod +x "${TARGET_PATH}"

PATH_EXPORT='export PATH="$HOME/.local/bin:$PATH"'

detect_rc_file() {
  local shell_name
  shell_name="$(basename "${SHELL:-}")"
  case "${shell_name}" in
    bash) echo "${HOME}/.bashrc" ;;
    zsh) echo "${HOME}/.zshrc" ;;
    *) echo "${HOME}/.profile" ;;
  esac
}

if [[ "${NO_PATH_UPDATE}" != "1" ]]; then
  RC_FILE="$(detect_rc_file)"
  touch "${RC_FILE}"
  if ! grep -Fqx "${PATH_EXPORT}" "${RC_FILE}"; then
    echo "${PATH_EXPORT}" >> "${RC_FILE}"
    echo "[install] Added ~/.local/bin to PATH in ${RC_FILE}"
  fi
fi

echo "[install] Installed command: ${TARGET_PATH}"
echo "[install] Verify with: autopipeline --help"
if [[ "${NO_PATH_UPDATE}" != "1" ]]; then
  echo "[install] If command not found, run: source \"${RC_FILE}\" or open a new shell."
else
  echo "[install] PATH was not updated (NO_PATH_UPDATE=1). Ensure ~/.local/bin is in PATH."
fi
