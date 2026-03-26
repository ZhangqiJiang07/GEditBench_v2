# Environment Profiles

This directory contains three publishable environment profiles:

- `annotate` (source env: `pipeline3.11`)
- `train` (source env: `ft`)
- `pvc_judge` (source env: `editscore`)

## Files

- `../pyproject.toml` + `../uv.lock`
  - Primary lock/install path for `uv sync --frozen`.
- `requirements/<profile>.lock.txt`
  - Python package lock file for `pip` / `uv`.
- `requirements/non_pypi/<profile>.txt`
  - Git/URL dependencies that are not available on PyPI (for example `sam-2`).
- `conda/<profile>.from-history.yml`
  - Conda minimal spec exported from install history (portable baseline).
- `conda/<profile>.explicit.txt`
  - Conda exact package URLs for strict reproduction on compatible systems.

## Refresh Exports

Run from project root:

```bash
./scripts/export_envs.sh
```

Or export one profile:

```bash
./scripts/export_envs.sh train
```

## Install

Unified installer:

```bash
./scripts/install.sh <annotate|train|pvc_judge> [options]
```

Examples:

```bash
# Auto-select manager: uv first, fallback to pip.
./scripts/install.sh annotate

# uv project lock mode (uses pyproject.toml + uv.lock if present)
# profile -> extra mapping: annotate->annotate, train->train, pvc_judge->pvc-judge

# Force pip
./scripts/install.sh train --manager pip

# Force uv and custom venv directory
./scripts/install.sh pvc_judge --manager uv --venv-dir ./.venvs/pvc_judge

# Skip non-PyPI phase if needed
./scripts/install.sh annotate --skip-non-pypi

# Use conda from-history
./scripts/install.sh train --manager conda --env-name train

# Use conda explicit lock (strict, platform-sensitive)
./scripts/install.sh pvc_judge --manager conda --conda-mode explicit --env-name pvc_judge
```

Direct uv usage:

```bash
uv sync --frozen --extra annotate
uv sync --frozen --extra train
uv sync --frozen --extra pvc-judge

# Non-PyPI dependencies are maintained separately:
uv pip install -r environments/requirements/non_pypi/annotate.txt
```
