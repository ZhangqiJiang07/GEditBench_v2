# Environment Profiles

This directory contains three publishable environment profiles:

- `annotate` (source env: `pipeline3.11`)
- `train` (source env: `ft`)
- `pvc_judge` (source env: `editscore`)

The exported manifests are normalized for public installation:

- profile-specific Python versions are recorded (`annotate` uses Python 3.11; `train` and `pvc_judge` use Python 3.12)
- PyTorch CUDA wheels are resolved from the official PyTorch wheel index
- installer-only packages (`pip`, `setuptools`, `wheel`, `uv`) are removed from the base manifests
- `annotate` no longer depends on an external `sam-2` Git checkout because SAM2 is vendored in-repo under `src/sam2`

## Files

- `requirements/<profile>.lock.txt`
  - Public pip/uv manifest for the profile.
- `requirements/optional/<profile>.txt`
  - Optional accelerator packages that are not required for a functional base install.
- `requirements/non_pypi/<profile>.txt`
  - External Git/URL dependencies. These are empty for the current public profiles.
- `conda/<profile>.from-history.yml`
  - Minimal conda bootstrap with the correct Python version and `pip`.
- `conda/<profile>.explicit.txt`
  - Exact conda package URLs for strict reproduction on compatible Linux systems.

## Refresh Exports

Run from the project root:

```bash
./scripts/export_envs.sh
```

Or export one profile:

```bash
./scripts/export_envs.sh train
```

## Install

Unified one-click installer:

```bash
./scripts/install.sh <annotate|train|pvc_judge> [options]
```

Examples:

```bash
# Recommended: uv if available, otherwise conda, then pip.
./scripts/install.sh annotate

# Create a conda env and then install the normalized pip manifest into it.
./scripts/install.sh train --manager conda --env-name train

# Force pip with an explicit interpreter.
./scripts/install.sh pvc_judge --manager pip --python-bin python3.12

# Install optional train accelerators such as flash-attn.
./scripts/install.sh train --with-optional
```

## Notes

- `train` defaults to the safer `sdpa` attention path in the checked-in training config. Install `requirements/optional/train.txt` if you want to enable FlashAttention again.
- The full `train` and `pvc_judge` manifests include large CUDA wheels; dry-run resolution can take a while even when there are no version conflicts.
