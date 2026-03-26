import json
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from common_utils.project_paths import CONFIGS_ROOT

DEFAULT_THRESHOLDS_CONFIG_FILE = str(CONFIGS_ROOT / "pipelines" / "data_construction_configs.json")
GPT_RESPONSE = '''```json
{{
    "winner": {winner_value}
}}
```
'''
FIXED_SEED = 42


def _normalize_tasks(tasks: str) -> List[str]:
    task_list = [task.strip() for task in tasks.split(",") if task.strip()]
    if not task_list:
        raise ValueError("No valid tasks found in `tasks`.")
    return task_list


def _resolve_data_save_path(output_dir: str, prefix: str) -> str:
    if prefix is None or prefix.strip() == "":
        return output_dir
    return os.path.join(output_dir, prefix.lstrip("/\\"))


def _load_jsonl(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _detect_mode(raw_data: List[dict]) -> str:
    if not raw_data:
        raise ValueError("Input grouped data is empty.")
    results = raw_data[0].get("results")
    if isinstance(results, list):
        return "group"
    if isinstance(results, dict):
        return "judge"
    raise ValueError("Unsupported grouped schema: `results` must be list or dict.")


def _convert_judge_mode(raw_data: List[dict], prompts_num: int, task: str) -> List[dict]:
    all_training_pairs = []
    recorded_image_instruction_pairs = set()

    for item in tqdm(raw_data, desc=f"Processing task {task}"):
        try:
            item_res = item["results"]
            input_dict = item_res["input_dict"]

            image_prompt_key = item["key"].split("_pair_")[0]
            recorded_image_instruction_pairs.add(image_prompt_key)

            source_image = input_dict.get("source_image", input_dict.get("input_image"))
            if source_image is None:
                raise KeyError("Missing source image field in input_dict")

            all_training_pairs.append(
                {
                    "edited_image_paths": input_dict["edited_images"],
                    "instruction": input_dict["instruction"],
                    "source_image_path": source_image,
                    "gpt_response": GPT_RESPONSE.format(winner_value=f"\"{item_res['winner']}\""),
                }
            )
        except Exception as e:
            print(f"[train-pairs][{task}] Error processing item: {e}")

        if len(recorded_image_instruction_pairs) >= prompts_num:
            break

    return all_training_pairs


def _get_metric_value(candidate: dict, area: str, metric: str):
    try:
        val = candidate.get(area, {}).get(metric, None)
        if val is None or val == "None" or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except (AttributeError, ValueError, TypeError):
        return None


def _clean_group_candidates(candidates: List[dict], area_configs: Dict) -> List[dict]:
    valid_candidates = []
    for cand in candidates:
        is_valid = True
        for area_name, config in area_configs.items():
            p_key = config["primary_key"]
            val = _get_metric_value(cand, area_name, p_key)
            if val is None:
                is_valid = False
                break
        if is_valid:
            valid_candidates.append(cand)
    return valid_candidates


def _safe_compare(val_a, val_b, direction):
    if val_a is None or val_b is None:
        return 0

    diff = direction * (val_a - val_b)
    if diff > 1e-6:
        return 1
    if diff < -1e-6:
        return -1
    return 0


def _construct_pairs(raw_candidates, area_configs: Dict):
    candidates = _clean_group_candidates(raw_candidates, area_configs)
    if len(candidates) < 2:
        return []
    df = pd.DataFrame(candidates)

    valid_z_count = 0
    total_z_scores = np.zeros(len(df))

    for area_name, config in area_configs.items():
        p_key = config["primary_key"]
        p_dir = config["primary_direction"]

        vals = df.apply(lambda row: _get_metric_value(row.to_dict(), area_name, p_key), axis=1).values
        vals = np.array(vals, dtype=float)
        directed_vals = vals * p_dir
        std = np.std(directed_vals)

        if std < 1e-6:
            df[f"z_{area_name}"] = 0.0
            continue

        z = (directed_vals - np.mean(directed_vals)) / std
        df[f"z_{area_name}"] = z
        total_z_scores += z
        valid_z_count += 1

    if valid_z_count == 0:
        return []

    df["global_z_score"] = total_z_scores / valid_z_count

    def assign_tier(row):
        is_failure = False
        is_elite = True
        for area_name, config in area_configs.items():
            z_val = row.get(f"z_{area_name}", 0.0)
            t_fail = config.get("threshold_failure", -0.5)
            t_elite = config.get("threshold_elite", 0.8)
            if z_val <= t_fail:
                is_failure = True
            if z_val <= t_elite:
                is_elite = False

        if is_failure:
            return "C"
        if is_elite:
            return "A"
        return "B"

    df["tier"] = df.apply(assign_tier, axis=1)
    tier_a = df[df["tier"] == "A"]
    tier_b = df[df["tier"] == "B"]
    tier_c = df[df["tier"] == "C"]

    raw_pairs = []
    strategies = [
        (tier_a, tier_c, "Elite_vs_Failure", 0.0),
        (tier_b, tier_c, "Normal_vs_Failure", 0.0),
        (tier_a, tier_b, "Elite_vs_Normal", 0.3),
    ]

    for winner_tier, loser_tier, pair_type, min_margin in strategies:
        for _, win in winner_tier.iterrows():
            for _, lose in loser_tier.iterrows():
                if (win["global_z_score"] - lose["global_z_score"]) < min_margin:
                    continue
                raw_pairs.append(
                    {
                        "chosen": win.to_dict(),
                        "rejected": lose.to_dict(),
                        "type": pair_type,
                    }
                )

    valid_pairs = []

    def check_robust_filters(win_row, lose_row):
        for area_name, config in area_configs.items():
            p_key = config["primary_key"]
            p_dir = config["primary_direction"]
            val_win = _get_metric_value(win_row, area_name, p_key)
            val_lose = _get_metric_value(lose_row, area_name, p_key)
            if _safe_compare(val_win, val_lose, p_dir) == -1:
                return False

        votes = 0
        valid_comparisons = 0
        total_constraints = 0
        for area_name, config in area_configs.items():
            constraints = config.get("secondary_constraints", {})
            total_constraints += len(constraints)
            if len(constraints) == 0:
                continue
            for metric, direction in constraints.items():
                val_win = _get_metric_value(win_row, area_name, metric)
                val_lose = _get_metric_value(lose_row, area_name, metric)

                result = _safe_compare(val_win, val_lose, direction)
                votes += result

                if val_win is not None and val_lose is not None:
                    valid_comparisons += 1

        if total_constraints == 0:
            return True
        if valid_comparisons == 0:
            return False
        return votes > 0

    for pair in raw_pairs:
        if check_robust_filters(pair["chosen"], pair["rejected"]):
            valid_pairs.append(pair)

    return valid_pairs


def _format_transfer(group_pairs: List[dict], rng: random.Random, filt_out_types=None):
    training_pairs = []
    for pair in group_pairs:
        if (filt_out_types is not None) and (pair["type"] in filt_out_types):
            continue

        if rng.random() > 0.5:
            training_pairs.append(
                {
                    "edited_image_paths": [pair["chosen"]["edited_image_path"], pair["rejected"]["edited_image_path"]],
                    "instruction": pair["chosen"]["instruction"],
                    "source_image_path": pair["chosen"]["source_image_path"],
                    "gpt_response": GPT_RESPONSE.format(winner_value="\"Image A\""),
                }
            )
        else:
            training_pairs.append(
                {
                    "edited_image_paths": [pair["rejected"]["edited_image_path"], pair["chosen"]["edited_image_path"]],
                    "instruction": pair["chosen"]["instruction"],
                    "source_image_path": pair["chosen"]["source_image_path"],
                    "gpt_response": GPT_RESPONSE.format(winner_value="\"Image B\""),
                }
            )
    return training_pairs


def _convert_group_mode(
    raw_data: List[dict],
    prompts_num: int,
    filt_out_strategy: str,
    area_configs: Dict,
    task: str,
    rng: random.Random,
) -> List[dict]:
    cleaned_data = []
    for raw_group in raw_data:
        try:
            cleaned_group = [item for item in raw_group["results"]]
            cleaned_data.append(cleaned_group)
        except Exception as e:
            print(f"[train-pairs][{task}] Error processing group: {e}")

    all_training_pairs = []
    task_prompts_num = 0
    for group in tqdm(cleaned_data, desc=f"Processing task {task}", total=len(cleaned_data)):
        group_pairs = _construct_pairs(group, area_configs)
        if filt_out_strategy == "head_tail":
            filt_out_types = ["Elite_vs_Normal", "Normal_vs_Failure"]
        elif filt_out_strategy == "three_tiers":
            filt_out_types = None
        else:
            raise ValueError(f"Unsupported filt_out_strategy: {filt_out_strategy}")

        training_pairs = _format_transfer(group_pairs, rng=rng, filt_out_types=filt_out_types)
        if len(training_pairs) < 1:
            continue

        all_training_pairs.extend(training_pairs)
        task_prompts_num += 1
        if task_prompts_num >= prompts_num:
            break

    return all_training_pairs


def convert_grouped_results_to_train_pairs(
    tasks: str,
    input_dir: str,
    output_dir: str,
    prompts_num: int = 1500,
    prefix: str = "",
    mode: str = "auto",
    filt_out_strategy: str = "three_tiers",
    thresholds_config_file: str = DEFAULT_THRESHOLDS_CONFIG_FILE,
) -> Dict[str, str]:
    tasks = _normalize_tasks(tasks)
    if mode not in {"auto", "group", "judge"}:
        raise ValueError(f"Unsupported mode: {mode}")

    data_save_path = _resolve_data_save_path(output_dir, prefix)
    os.makedirs(data_save_path, exist_ok=True)

    task_metrics_config_map = {}
    if mode in {"auto", "group"}:
        with open(thresholds_config_file, "r", encoding="utf-8") as f:
            task_metrics_config_map = json.load(f)

    rng = random.Random(FIXED_SEED)
    output_paths = {}

    for task in tasks:
        input_task_json = os.path.join(input_dir, f"{task}_grouped.jsonl")
        if not os.path.exists(input_task_json):
            raise FileNotFoundError(f"Input file not found: {input_task_json}")

        raw_data = _load_jsonl(input_task_json)
        resolved_mode = _detect_mode(raw_data) if mode == "auto" else mode

        if resolved_mode == "group":
            area_configs = task_metrics_config_map.get(task, None)
            if area_configs is None:
                raise KeyError(f"Config for task `{task}` is missing in {thresholds_config_file}")
            all_training_pairs = _convert_group_mode(
                raw_data=raw_data,
                prompts_num=prompts_num,
                filt_out_strategy=filt_out_strategy,
                area_configs=area_configs,
                task=task,
                rng=rng,
            )
        elif resolved_mode == "judge":
            all_training_pairs = _convert_judge_mode(
                raw_data=raw_data,
                prompts_num=prompts_num,
                task=task,
            )
        else:
            raise ValueError(f"Unsupported resolved mode: {resolved_mode}")

        output_task_json = os.path.join(data_save_path, f"{task}.json")
        with open(output_task_json, "w", encoding="utf-8") as f_out:
            json.dump(all_training_pairs, f_out, indent=4, ensure_ascii=False)

        output_paths[task] = output_task_json

    return output_paths
