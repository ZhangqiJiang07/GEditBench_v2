#!/usr/bin/env bash
set -euo pipefail

TARGET_PATH="${HOME}/.local/bin/autogen"

if [[ -f "${TARGET_PATH}" ]]; then
  rm -f "${TARGET_PATH}"
  echo "[uninstall] Removed ${TARGET_PATH}"
else
  echo "[uninstall] Command not found at ${TARGET_PATH}"
fi

echo "[uninstall] If needed, remove PATH line from your shell rc:"
echo '  export PATH="$HOME/.local/bin:$PATH"'
