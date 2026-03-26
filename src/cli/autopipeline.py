import os
import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common_utils.logging_util import logger_init
from src.common_utils.project_paths import CONFIGS_ROOT, DATA_ROOT, LOGS_ROOT, normalize_benchmark_name

_LOGGER = None

DEFAULT_CONFIG_BASE_DIR = str(CONFIGS_ROOT / "pipelines")
DEFAULT_USER_CONFIG = str(CONFIGS_ROOT / "pipelines" / "user_config.yaml")
DEFAULT_CANDIDATE_POOL_DIR = str(CONFIGS_ROOT / "datasets" / "candidate_pools")
DEFAULT_BMK_CONFIG = str(CONFIGS_ROOT / "datasets" / "bmk.json")
DEFAULT_ANNOTATION_SAVE_PATH = str(DATA_ROOT / "c_annotated_group_data")
DEFAULT_EVAL_SAVE_PATH = str(DATA_ROOT / "reward_eval_results")
DEFAULT_TRAIN_PAIRS_INPUT_DIR = str(DATA_ROOT / "c_annotated_group_data")
DEFAULT_TRAIN_PAIRS_OUTPUT_DIR = str(DATA_ROOT / "d_train_data")
DEFAULT_TRAIN_PAIRS_THRESHOLDS_CONFIG = str(CONFIGS_ROOT / "pipelines" / "data_construction_configs.json")


def _get_logger():
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logger_init(
            log_dir=str(LOGS_ROOT / "pipeline_runs"),
            level="info",
            # main_process_only=True,
        )
    return _LOGGER


def _pipeline_tag(pipeline_name: str) -> str:
    _temp_name = pipeline_name.replace("/", "_").replace(".", "_").lower()
    if _temp_name[0] == '_':
        _temp_name = _temp_name[1:]
    return _temp_name


def _build_executor(pipeline_name: str, max_workers: int):
    from src.autopipeline.runners import ProcessExecutor, ThreadExecutor
    if pipeline_name in ["vlm-as-a-judge"]:
        return ThreadExecutor(max_workers=max_workers)
    return ProcessExecutor(max_workers=max_workers)


def _write_annotation_results(all_results: Dict[str, Dict[str, Any]], pipeline_name: str, results_file: str):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        if pipeline_name in ["human-centric", "object-centric"]:
            grouped_res_dict = {}
            for item_key, res in all_results.items():
                source_image_key = item_key.rsplit("_model_", 1)[0]
                grouped_res_dict.setdefault(source_image_key, []).append(res["output"])

            for item_key, res_list in grouped_res_dict.items():
                res_line = {
                    "key": item_key,
                    "results": res_list,
                }
                f.write(json.dumps(res_line, ensure_ascii=False) + "\n")
        else:
            for item_key, item_res_dict in all_results.items():
                if item_res_dict.get("winner") in ["Failed", "Tie"]:
                    continue
                res_line = {
                    "key": item_key,
                    "results": item_res_dict,
                }
                f.write(json.dumps(res_line, ensure_ascii=False) + "\n")


def _write_eval_results(all_results: Dict[str, Dict[str, Any]], results_file: str):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        for item_key, item_res_dict in all_results.items():
            if item_res_dict.get("winner") in ["Failed", "Tie"]:
                continue
            res_line = {
                "key": item_key,
                "results": item_res_dict,
            }
            f.write(json.dumps(res_line, ensure_ascii=False) + "\n")


def run_annotation(
    edit_task: str,
    pipeline_config_path: str,
    max_workers: int = 4,
    save_path: str = DEFAULT_ANNOTATION_SAVE_PATH,
    user_config: str = DEFAULT_USER_CONFIG,
    candidate_pool_dir: str = DEFAULT_CANDIDATE_POOL_DIR,
):
    from common_utils.dataset_loader import CandidatesDataset
    from src.autopipeline.pipelines import PipelineLoader
    from src.autopipeline.runners import AnnotatorWorker, Runner
    from src.core.cache_manager import CacheManager
    from src.core.config_engine import ConfigEngine

    _get_logger().info(f"Starting annotation for edit task `{edit_task}` with pipeline config `{pipeline_config_path}`")
    edit_task = edit_task.replace("-", "_").lower()
    candidate_pool_path = os.path.join(candidate_pool_dir, f"{edit_task}.json")
    if not os.path.exists(candidate_pool_path):
        raise FileNotFoundError(f"Candidate pool config not found: {candidate_pool_path}")

    with open(candidate_pool_path, "r", encoding="utf-8") as f:
        data_config = json.load(f)

    engine = ConfigEngine()
    pipeline_config = engine.load(
        pipeline_path=pipeline_config_path,
        user_path=user_config,
    )
    _get_logger().info(f"Pipeline config loaded successfully from {pipeline_config_path}. Config content: {pipeline_config}")

    pipeline_name = pipeline_config["name"]
    support_task = pipeline_config.get("support_task")
    if support_task and edit_task not in support_task:
        raise ValueError(
            f"Pipeline `{pipeline_name}` does not support task `{edit_task}`. "
            f"Supported tasks: {support_task}"
        )

    cache_path = os.path.join(
        save_path,
        ".cache",
        f"{edit_task}_{_pipeline_tag(pipeline_name)}_results_cache.jsonl",
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    runner = Runner(
        pipeline_loader=PipelineLoader(pipeline_config),
        worker=AnnotatorWorker(pipeline_name, edit_task),
        executor=_build_executor(pipeline_name, max_workers=max_workers),
        cache_manager=CacheManager(cache_path),
        dataset=CandidatesDataset(data_config, pipeline_name),
    )

    all_results = runner.run()

    results_file = os.path.join(save_path, f"{edit_task}_grouped.jsonl")
    _get_logger().info("Saving final grouped results to %s...", results_file)
    _write_annotation_results(all_results, pipeline_name, results_file)
    return results_file


def run_eval(
    bmk: str,
    pipeline_config_path: str,
    max_workers: int = 4,
    save_path: str = DEFAULT_EVAL_SAVE_PATH,
    user_config: str = DEFAULT_USER_CONFIG,
    bmk_config: str = DEFAULT_BMK_CONFIG,
    geditv2_metadata_file: str = None,
):
    import pandas as pd
    from common_utils.dataset_loader import load_dataset
    from src.autopipeline.pipelines import PipelineLoader
    from src.autopipeline.runners import EvalWorker, Runner
    from src.core.cache_manager import CacheManager
    from src.core.config_engine import ConfigEngine
    normalized_bmk = normalize_benchmark_name(bmk)
    if normalized_bmk == "geditv2" and geditv2_metadata_file is None:
        print("geditv2_metadata_file is not specified. Defaulting to 'metadata.jsonl'.")
        geditv2_metadata_file = "metadata.jsonl"

    _get_logger().info(
        "Starting evaluation for benchmark `%s` (resolved as `%s`) with pipeline config `%s`",
        bmk,
        normalized_bmk,
        pipeline_config_path,
    )
    if not os.path.exists(bmk_config):
        raise FileNotFoundError(f"Reward benchmark config not found: {bmk_config}")

    with open(bmk_config, "r", encoding="utf-8") as f:
        bmk_config = json.load(f)

    if normalized_bmk not in bmk_config:
        raise KeyError(
            f"Benchmark `{normalized_bmk}` not found in {bmk_config}. Available benchmarks: {list(bmk_config.keys())}"
        )
    bmk_info = dict(bmk_config[normalized_bmk])
    bmk_info.setdefault("max_workers", max_workers)
    bmk_info.setdefault("meta_file", geditv2_metadata_file)

    engine = ConfigEngine()
    pipeline_config = engine.load(
        pipeline_path=pipeline_config_path,
        user_path=user_config,
    )

    pipeline_name = pipeline_config["name"]
    cache_path = os.path.join(
        save_path,
        ".cache",
        f"{normalized_bmk}_{_pipeline_tag(pipeline_config_path)}_results_cache.jsonl",
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    runner = Runner(
        pipeline_loader=PipelineLoader(pipeline_config),
        worker=EvalWorker(pipeline_name),
        executor=_build_executor(pipeline_name, max_workers=max_workers),
        cache_manager=CacheManager(cache_path),
        dataset=load_dataset(bmk_name=normalized_bmk, **bmk_info),
    )

    all_results = runner.run()
    eval_res_name = os.path.splitext(os.path.basename(pipeline_config_path))[0]
    if normalized_bmk == "geditv2":
        eval_res_name += f"_{os.path.splitext(os.path.basename(geditv2_metadata_file))[0]}"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_save_path = os.path.join(save_path, normalized_bmk, eval_res_name)
    os.makedirs(results_save_path, exist_ok=True)
    results_file = os.path.join(results_save_path, f"{timestamp}.jsonl")
    _get_logger().info("Saving final grouped results to %s...", results_file)
    _write_eval_results(all_results, results_file)
    return results_file


def run_train_pairs(
    tasks: str,
    prompts_num: int = 1500,
    prefix: str = "",
    input_dir: str = DEFAULT_TRAIN_PAIRS_INPUT_DIR,
    output_dir: str = DEFAULT_TRAIN_PAIRS_OUTPUT_DIR,
    mode: str = "auto",
    filt_out_strategy: str = "three_tiers",
    thresholds_config_file: str = DEFAULT_TRAIN_PAIRS_THRESHOLDS_CONFIG,
):
    from src.autopipeline.postprocess.train_pairs import convert_grouped_results_to_train_pairs

    _get_logger().info(
        "Starting train-pairs conversion. tasks=%s, mode=%s, input_dir=%s, output_dir=%s",
        tasks,
        mode,
        input_dir,
        output_dir,
    )
    output_paths = convert_grouped_results_to_train_pairs(
        tasks=tasks,
        input_dir=input_dir,
        output_dir=output_dir,
        prompts_num=prompts_num,
        prefix=prefix,
        mode=mode,
        filt_out_strategy=filt_out_strategy,
        thresholds_config_file=thresholds_config_file,
    )
    _get_logger().info("Train-pairs conversion finished. Output files: %s", output_paths)
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autopipeline",
        description="Unified CLI for autopipeline annotation, evaluation, and train-pair conversion.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    anno_parser = subparsers.add_parser("annotation", help="Run annotation pipeline.")
    anno_parser.add_argument("--edit-task", type=str, required=True, help="Edit task name.")
    anno_parser.add_argument(
        "--pipeline-config-path",
        type=str,
        required=True,
        help="Pipeline config path (absolute).",
    )
    anno_parser.add_argument("--max-workers", type=int, default=4, help="Max worker count.")
    anno_parser.add_argument(
        "--save-path",
        type=str,
        default=DEFAULT_ANNOTATION_SAVE_PATH,
        help="Directory to save outputs and cache.",
    )
    anno_parser.add_argument(
        "--user-config",
        type=str,
        default=DEFAULT_USER_CONFIG,
        help="User config YAML path (absolute).",
    )
    anno_parser.add_argument(
        "--candidate-pool-dir",
        type=str,
        default=DEFAULT_CANDIDATE_POOL_DIR,
        help="Directory containing candidate pool JSON files.",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation pipeline.")
    eval_parser.add_argument("--bmk", type=str, required=True, help="Benchmark key in reward_bmk.json.")
    eval_parser.add_argument(
        "--pipeline-config-path",
        type=str,
        required=True,
        help="Pipeline config path (absolute).",
    )
    eval_parser.add_argument("--max-workers", type=int, default=4, help="Max worker count.")
    eval_parser.add_argument(
        "--save-path",
        type=str,
        default=DEFAULT_EVAL_SAVE_PATH,
        help="Directory to save outputs and cache.",
    )
    eval_parser.add_argument(
        "--user-config",
        type=str,
        default=DEFAULT_USER_CONFIG,
        help="User config YAML path (absolute).",
    )
    eval_parser.add_argument(
        "--bmk-config",
        type=str,
        default=DEFAULT_BMK_CONFIG,
        help="Benchmark configuration JSON path.",
    )
    eval_parser.add_argument(
        "--geditv2-metadata-file",
        "--openedit-metadata-file",
        dest="geditv2_metadata_file",
        type=str,
        default=None,
        help="Metadata JSONL file name under the benchmark path (used for GEditBench v2 evaluation).",
    )

    train_pairs_parser = subparsers.add_parser("train-pairs", help="Convert grouped annotation results to train pairs.")
    train_pairs_parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated task names, e.g. color_alter,material_alter.",
    )
    train_pairs_parser.add_argument(
        "--prompts-num",
        type=int,
        default=1500,
        help="Number of prompts/groups to process for each task.",
    )
    train_pairs_parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional output subfolder under --output-dir.",
    )
    train_pairs_parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_TRAIN_PAIRS_INPUT_DIR,
        help="Directory containing <task>_grouped.jsonl files.",
    )
    train_pairs_parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_TRAIN_PAIRS_OUTPUT_DIR,
        help="Directory to save converted train-pair JSON files.",
    )
    train_pairs_parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "group", "judge"],
        default="auto",
        help="Conversion mode. auto detects schema from grouped results.",
    )
    train_pairs_parser.add_argument(
        "--filt-out-strategy",
        type=str,
        choices=["head_tail", "three_tiers"],
        default="three_tiers",
        help="Pair filtering strategy for group-mode conversion.",
    )
    train_pairs_parser.add_argument(
        "--thresholds-config-file",
        type=str,
        default=DEFAULT_TRAIN_PAIRS_THRESHOLDS_CONFIG,
        help="Task metric thresholds JSON (used in group mode).",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "annotation":
        run_annotation(
            edit_task=args.edit_task,
            pipeline_config_path=args.pipeline_config_path,
            max_workers=args.max_workers,
            save_path=args.save_path,
            user_config=args.user_config,
            candidate_pool_dir=args.candidate_pool_dir,
        )
    elif args.command == "eval":
        run_eval(
            bmk=args.bmk,
            pipeline_config_path=args.pipeline_config_path,
            max_workers=args.max_workers,
            save_path=args.save_path,
            user_config=args.user_config,
            bmk_config=args.bmk_config,
            geditv2_metadata_file=args.geditv2_metadata_file,
        )
    elif args.command == "train-pairs":
        run_train_pairs(
            tasks=args.tasks,
            prompts_num=args.prompts_num,
            prefix=args.prefix,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            filt_out_strategy=args.filt_out_strategy,
            thresholds_config_file=args.thresholds_config_file,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
