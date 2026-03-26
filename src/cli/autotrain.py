import argparse
import datetime as dt
import os
import shlex
import subprocess

from common_utils.project_paths import CONFIGS_ROOT, OUTPUT_ROOT, PROJECT_ROOT


DEFAULT_CONFIG_PATH = str(CONFIGS_ROOT / "lora_sft")
DEFAULT_OUTPUT_BASE = str(OUTPUT_ROOT / "sft_runs")
DEFAULT_TRAIN_SCRIPT = str(PROJECT_ROOT / "src" / "autotrain" / "train" / "train_sft_lora.py")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotrain",
        description="Launch SFT LoRA training with deepspeed.",
    )
    parser.add_argument("--config", required=True, help="Config file stem under --config-path, e.g. qwen3_sft")
    parser.add_argument("--num-gpus", required=True, type=int, help="Number of GPUs for deepspeed --num_gpus")
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Directory containing YAML config files.",
    )
    parser.add_argument(
        "--output-dir-base",
        default=DEFAULT_OUTPUT_BASE,
        help="Base output directory (date folder MMDD is appended automatically).",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root used as subprocess working directory.",
    )
    parser.add_argument(
        "--train-script",
        default=DEFAULT_TRAIN_SCRIPT,
        help="Path to the training entry script.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name suffix. Defaults to run_HHMMSS.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Overrides auto naming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command and paths without launching training.",
    )
    return parser


def _append_pythonpath(env: dict, src_path: str) -> None:
    py_path = env.get("PYTHONPATH", "")
    entries = [entry for entry in py_path.split(os.pathsep) if entry]
    if src_path not in entries:
        env["PYTHONPATH"] = src_path if not entries else f"{src_path}{os.pathsep}{py_path}"


def _resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    config = os.path.join(args.config_path, f"{args.config}.yaml")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        day = dt.datetime.now().strftime("%m%d")
        run_name = args.run_name or f"run_{dt.datetime.now().strftime('%H%M%S')}"
        output_dir = os.path.join(
            args.output_dir_base,
            day,
            f"{args.config}_{run_name}",
        )
    return config, output_dir


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.num_gpus <= 0:
        parser.error("num_gpus must be a positive integer")

    config, output_dir = _resolve_paths(args)
    if not os.path.isfile(config):
        parser.error(f"Config not found: {config}")

    triton_cache_dir = os.path.join("/tmp", os.environ.get("USER", "unknown"), "triton_cache")

    cmd = [
        "deepspeed",
        "--num_gpus",
        str(args.num_gpus),
        args.train_script,
        "--config",
        config,
        "--output_dir",
        output_dir,
    ]

    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = triton_cache_dir
    src_path = os.path.join(args.project_root, "src")
    _append_pythonpath(env, src_path)

    print("************************************************************")
    print("Starting VLM Finetuning with DeepSpeed")
    print("************************************************************")
    print(f"Configuration: {config}")
    print(f"Output Directory: {output_dir}")
    print(f"TRITON_CACHE_DIR: {triton_cache_dir}")
    print(f"PYTHONPATH: {env.get('PYTHONPATH', '')}")
    print("Command:", shlex.join(cmd))
    print("------------------------------------------------------------")

    if args.dry_run:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(triton_cache_dir, exist_ok=True)

    result = subprocess.run(cmd, cwd=args.project_root, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
