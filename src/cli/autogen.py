import argparse
import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autogen.constants import QWEN3_VL_EMBEDDING_MODEL_PATH
from common_utils.project_paths import CONFIGS_ROOT, DATA_ROOT, PROJECT_ROOT, normalize_benchmark_name

DEFAULT_FILTER_INPUT_FILE = str(DATA_ROOT / "a_raw_img_prompt_pair_data" / "subject_add.jsonl")
DEFAULT_FILTER_OUTPUT_DIR = str(DATA_ROOT / "b_filtered_img_prompt_pair_data")
DEFAULT_QWEN_EMBEDDING_MODEL_PATH = QWEN3_VL_EMBEDDING_MODEL_PATH
DEFAULT_DATASET_PATH = str(DATA_ROOT / "b_filtered_img_prompt_pair_data")
DEFAULT_GEDITV2_IMAGE_SAVE_DIR = str(PROJECT_ROOT / "geditv2_bench" / "images" / "edited")
DEFAULT_BMK_CONFIG_PATH = str(CONFIGS_ROOT / "datasets" / "bmk.json")


def _append_pythonpath_env(src_path: str) -> None:
    py_path = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in py_path.split(os.pathsep) if entry]
    if src_path not in entries:
        os.environ["PYTHONPATH"] = src_path if not entries else f"{src_path}{os.pathsep}{py_path}"


def _bootstrap_import_paths() -> None:
    src_root = str(SRC_ROOT)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    _append_pythonpath_env(src_root)


def _configure_mp_runtime() -> None:
    import torch.multiprocessing as mp

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mp.set_start_method("spawn", force=True)


def _cmd_filter(args: argparse.Namespace) -> int:
    from autogen.workflows.filtering import filter_dataset

    output_file = filter_dataset(
        sample_num=args.sample_num,
        task=args.task,
        input_file=args.input_file,
        output_dir=args.output_dir,
        qwen_embedding_model_path=args.qwen_embedding_model_path,
        image_save_path=args.image_save_path,
        embedding_batch_size=args.embedding_batch_size,
    )
    print(f"Finished. Output saved to: {output_file}")
    return 0


def _cmd_run_candidates(args: argparse.Namespace) -> int:
    from autogen.workflows.generation import run_dataset_generation

    _configure_mp_runtime()
    output_file = run_dataset_generation(
        task=args.task,
        model=args.model,
        dataset_path=args.dataset_path,
        gpus_per_worker=args.gpus_per_worker,
        output_bucket_prefix=args.output_bucket_prefix,
    )
    print(f"Finished. Results saved to: {output_file}")
    return 0


def _cmd_run_geditv2(args: argparse.Namespace) -> int:
    from autogen.workflows.generation import run_geditv2_generation

    _configure_mp_runtime()
    output_file, merged_output = run_geditv2_generation(
        model=args.model,
        gpus_per_worker=args.gpus_per_worker,
        bench_path=args.bench_path,
        image_save_dir=args.image_save_dir,
        merge_to_metadata=args.merge_to_metadata,
        bmk_config_path=args.bmk_config_path,
    )
    print(f"Finished. Results saved to: {output_file}")
    if merged_output is not None:
        print(f"Merged metadata saved to: {merged_output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autogen",
        description="Unified CLI for autogen filtering and generation workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter", help="Filter prepared data with Qwen3-VL embeddings.")
    filter_parser.add_argument("--sample-num", type=int, default=1500)
    filter_parser.add_argument("--task", type=str, default="background_change")
    filter_parser.add_argument("--input-file", type=str, default=DEFAULT_FILTER_INPUT_FILE)
    filter_parser.add_argument("--output-dir", type=str, default=DEFAULT_FILTER_OUTPUT_DIR)
    filter_parser.add_argument("--qwen-embedding-model-path", type=str, default=DEFAULT_QWEN_EMBEDDING_MODEL_PATH)
    filter_parser.add_argument("--image-save-path", type=str, default=None)
    filter_parser.add_argument("--embedding-batch-size", type=int, default=256)
    filter_parser.set_defaults(func=_cmd_filter)

    run_parser = subparsers.add_parser("run", help="Run generation workflows.")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)

    run_candidates_parser = run_subparsers.add_parser("candidates", help="Run candidates generation.")
    run_candidates_parser.add_argument("--task", type=str, default="subject-add")
    run_candidates_parser.add_argument("--model", type=str, default="qwen-image-edit")
    run_candidates_parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    run_candidates_parser.add_argument("--gpus-per-worker", type=int, default=1)
    run_candidates_parser.add_argument("--output-bucket-prefix", type=str, default="data/generated_images")
    run_candidates_parser.set_defaults(func=_cmd_run_candidates)

    for command_name, help_text in (
        ("geditv2", "Run GEditBench v2 benchmark generation."),
        ("openedit", argparse.SUPPRESS),
    ):
        run_geditv2_parser = run_subparsers.add_parser(command_name, help=help_text)
        run_geditv2_parser.add_argument("--model", type=str, default="qwen-image-edit")
        run_geditv2_parser.add_argument("--gpus-per-worker", type=int, default=1)
        run_geditv2_parser.add_argument("--bench-path", type=str, default=None)
        run_geditv2_parser.add_argument("--image-save-dir", type=str, default=DEFAULT_GEDITV2_IMAGE_SAVE_DIR)
        run_geditv2_parser.add_argument("--merge-to-metadata", action="store_true")
        run_geditv2_parser.add_argument("--bmk-config-path", type=str, default=DEFAULT_BMK_CONFIG_PATH)
        run_geditv2_parser.set_defaults(func=_cmd_run_geditv2, benchmark_name=normalize_benchmark_name(command_name))

    return parser


def main(argv=None) -> int:
    _bootstrap_import_paths()
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
