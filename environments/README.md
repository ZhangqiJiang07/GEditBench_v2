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

- `<profile>.yml`
  - Direct-install conda manifest. Use `conda env create -f environments/<profile>.yml`.
- `requirements/<profile>.lock.txt`
  - Direct pip manifest for the profile.
- `requirements/optional/<profile>.txt`
  - Optional accelerator packages that are not required for a functional base install.
- `requirements/non_pypi/<profile>.txt`
  - External Git/URL dependencies. These are empty for the current public profiles and are already referenced by `<profile>.yml`.
- `conda/<profile>.from-history.yml`
  - Minimal conda bootstrap used to generate `<profile>.yml`.
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

Use the static manifests directly.

```bash
conda env create -f environments/annotate.yml
conda activate annotate
```

Or use `venv + pip`:

```bash
python3.11 -m venv .venvs/annotate
source .venvs/annotate/bin/activate
python -m pip install -r environments/requirements/annotate.lock.txt
```

Profile-specific examples:

```bash
# annotate
conda env create -f environments/annotate.yml
# or:
python3.11 -m venv .venvs/annotate
source .venvs/annotate/bin/activate
python -m pip install -r environments/requirements/annotate.lock.txt

# train
conda env create -f environments/train.yml
# or:
python3.12 -m venv .venvs/train
source .venvs/train/bin/activate
python -m pip install -r environments/requirements/train.lock.txt
python -m pip install -r environments/requirements/optional/train.txt

# pvc_judge
conda env create -f environments/pvc_judge.yml
# or:
python3.12 -m venv .venvs/pvc_judge
source .venvs/pvc_judge/bin/activate
python -m pip install -r environments/requirements/pvc_judge.lock.txt
```

`scripts/install.sh` is still available, but it is now only a thin wrapper over these checked-in manifests.

## Notes

- `train` defaults to the safer `sdpa` attention path in the checked-in training config. Install `requirements/optional/train.txt` if you want to enable FlashAttention again.
- The full `train` and `pvc_judge` manifests include large CUDA wheels; dry-run resolution can take a while even when there are no version conflicts.
