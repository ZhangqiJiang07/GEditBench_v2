# Installation

Recommended runtime prerequisites:

- Python `3.11`
- Valid model paths, local service endpoints, or API backends
- Node.js `>= 20` if you also plan to build this documentation site

## Recommended installation path

The repository already provides a unified installation script. That should be your default starting point.

```bash
cd <PROJECT_ROOT>
./scripts/install.sh annotate
```

This command tries `uv` first and falls back to other package managers if needed. For the full environment matrix, see `environments/README.md`.

## Install the CLI wrapper

If you want to call `autopipeline` directly instead of typing `python -m src.cli.autopipeline` every time, run:

```bash
cd <PROJECT_ROOT>
./scripts/install_autopipeline.sh
autopipeline --help
```

## Manual installation

If you want explicit control over the virtual environment, you can install with `uv`:

```bash
cd <PROJECT_ROOT>
uv sync --frozen --extra annotate
uv pip install -r environments/requirements/non_pypi/annotate.txt
```

If you prefer `pip` or `conda`, the same script supports both:

```bash
./scripts/install.sh annotate --manager pip
./scripts/install.sh annotate --manager conda --env-name annotate
```

## User configuration

AutoPipeline depends heavily on `configs/pipelines/user_config.yaml`. At minimum, verify these two sections:

- `client_config`
  Model service endpoints, API keys, ports, timeouts, and retry settings.
- `model_paths`
  Local model paths for CLIP, DINO, SAM, ArcFace, Depth Anything, and related assets.

Recommended practice:

1. Do not commit real API keys to a public repository.
2. Maintain a private local override or a machine-specific copy of the config.
3. Make sure every `model_paths` entry points to a real location on the current machine.

## Additional model preparation

It is a good idea to download the InsightFace assets before your first multi-process run, so workers do not race on first-time initialization:

```bash
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
```

## Verify the installation

At minimum, validate these three entry points:

```bash
autopipeline --help
python -m src.cli.autopipeline --help
python -c "import src.autopipeline"
```

## Documentation site note

If you are maintaining the documentation site itself rather than the runtime, Docusaurus requires Node.js `>= 20`. The host environment in this workspace still defaults to Node 16, so local builds should use a Node 20 runner.

If you want the shortest path to a successful runtime command, continue with [Quickstart](/docs/getting-started/quickstart-autopipeline).
