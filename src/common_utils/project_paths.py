import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
LOGS_ROOT = PROJECT_ROOT / "logs"
OUTPUT_ROOT = PROJECT_ROOT / "output"
MODEL_ZOO_ROOT = Path(os.environ.get("GEDITV2_MODEL_ZOO_ROOT", "/path/to/model-zoo")).expanduser()
DEFAULT_BENCHMARK_NAME = "geditv2"
REMOTE_PATH_PREFIXES = ("s3://", "http://", "https://")

BENCHMARK_ALIASES = {
    "openedit": DEFAULT_BENCHMARK_NAME,
    DEFAULT_BENCHMARK_NAME: DEFAULT_BENCHMARK_NAME,
}


def normalize_benchmark_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return name
    return BENCHMARK_ALIASES.get(name.lower(), name)


def resolve_project_path(path: str | os.PathLike | None) -> str | Path | None:
    if path is None:
        return None
    path_str = str(path)
    if path_str.startswith(REMOTE_PATH_PREFIXES):
        return path_str
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def resolve_model_zoo_path(relative_path: str, env_var: str | None = None) -> Path:
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return Path(override).expanduser()
    return MODEL_ZOO_ROOT / relative_path
