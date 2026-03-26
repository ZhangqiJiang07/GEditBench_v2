import os
import json
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from typing import List, Dict

from common_utils.data_construction_configs import TASK_METRICS_CONFIG_MAP

PROMPTS_NUM = 1700
INPUT_JSON_PATH = "/data/auto_pipeline/results/scale_up_pair_results"
GPT_RESPONSE = '''```json
{{
    "winner": {winner_value}
}}
```
'''

random.seed(42)

def get_metric_value(candidate: dict, area: str, metric: str):
    """安全获取嵌套的 metric 值"""
    try:
        val = candidate.get(area, {}).get(metric, None)
        # 额外检查：有些系统可能返回字符串 "None" 或 NaN
        if val is None or val == "None" or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except (AttributeError, ValueError, TypeError):
        return None

def clean_group_candidates(candidates: List[dict], area_configs: Dict) -> List[dict]:
    """
    【核心机制 1】: 强力清洗
    剔除任何在 Primary Key 上为 None 的样本。
    """
    valid_candidates = []
    for cand in candidates:
        is_valid = True
        # 检查每一个区域的 Primary Key
        for area_name, config in area_configs.items():
            p_key = config['primary_key']
            val = get_metric_value(cand, area_name, p_key)
            
            # 如果主要指标缺失，标记为无效样本
            if val is None:
                is_valid = False
                break
        
        if is_valid:
            valid_candidates.append(cand)
            
    return valid_candidates

def safe_compare(val_a, val_b, direction):
    """
    【核心机制 2】: 鲁棒比较
    返回: 
       1 (A 优于 B)
      -1 (A 劣于 B)
       0 (平局 或 数据缺失)
    """
    if val_a is None or val_b is None:
        return 0
    
    diff = direction * (val_a - val_b)
    if diff > 1e-6: # 浮点数容差
        return 1
    elif diff < -1e-6:
        return -1
    else:
        return 0

def construct_pairs(raw_candidates, area_configs: Dict):
    candidates = clean_group_candidates(raw_candidates, area_configs)
    if len(candidates) < 2:
        return []
    df = pd.DataFrame(candidates)
    
    # --- Step 1: 计算每个区域的独立 Z-Score ---
    valid_z_count = 0
    total_z_scores = np.zeros(len(df))

    for area_name, config in area_configs.items():
        p_key = config['primary_key']
        p_dir = config['primary_direction']
        
        vals = df.apply(lambda row: get_metric_value(row.to_dict(), area_name, p_key), axis=1).values
        vals = np.array(vals, dtype=float)
        
        directed_vals = vals * p_dir
        std = np.std(directed_vals)
        
        if std < 1e-6:
            df[f'z_{area_name}'] = 0.0 # 方差太小，Z-Score 为 0
            continue
            
        z = (directed_vals - np.mean(directed_vals)) / std
        df[f'z_{area_name}'] = z
        
        total_z_scores += z
        valid_z_count += 1
    
    if valid_z_count == 0:
        return []

    df['global_z_score'] = total_z_scores / valid_z_count
    
    # --- Step 2: 基于各区域独立阈值的 "木桶效应" Tier 划分 ---
    def assign_tier(row):
        is_failure = False
        is_elite = True
        
        for area_name, config in area_configs.items():
            z_val = row.get(f'z_{area_name}', 0.0)
            t_fail = config.get('threshold_failure', -0.5)
            t_elite = config.get('threshold_elite', 0.8)
            
            # 1. 一票否决：任何区域低于该区域 Failure 阈值
            if z_val <= t_fail:
                is_failure = True
                
            # 2. 完美要求：必须所有区域都高于 Elite 阈值
            if z_val <= t_elite:
                is_elite = False
                
        if is_failure:
            return 'C' # Failure
        elif is_elite:
            return 'A' # Elite
        else:
            return 'B' # Normal
    
    # 2. 划分 Tier (基于你的分布图)
    df['tier'] = df.apply(assign_tier, axis=1)
    tier_a = df[df['tier'] == 'A']
    tier_b = df[df['tier'] == 'B']
    tier_c = df[df['tier'] == 'C']
    
    raw_pairs = []
    strategies = [
        (tier_a, tier_c, 'Elite_vs_Failure', 0.0),
        (tier_b, tier_c, 'Normal_vs_Failure', 0.0),
        (tier_a, tier_b, 'Elite_vs_Normal', 0.3) # 这里的 0.3 依然参考 global_z_score 的差值，防止边界摩擦
    ]

    for winner_tier, loser_tier, pair_type, min_margin in strategies:
        for _, win in winner_tier.iterrows():
            for _, lose in loser_tier.iterrows():
                # 依然用 global_z_score 计算 margin，确保宏观上赢家确实比输家分高
                if (win['global_z_score'] - lose['global_z_score']) < min_margin:
                    continue
                raw_pairs.append({
                    'chosen': win.to_dict(),
                    'rejected': lose.to_dict(),
                    'type': pair_type
                })
    
    valid_pairs = []

    # --- Step 3: 鲁棒的过滤逻辑 ---
    
    def check_robust_filters(win_row, lose_row):
        
        # A. Primary Key Strict Pareto
        # 规则：Winner 在任何 Primary Key 上都不能 "显著输掉" (-1)
        # 如果是 None (0)，我们容忍（Benefit of the doubt）
        for area_name, config in area_configs.items():
            p_key = config['primary_key']
            p_dir = config['primary_direction']
            
            val_win = get_metric_value(win_row, area_name, p_key)
            val_lose = get_metric_value(lose_row, area_name, p_key)
            
            # 使用 safe_compare
            # 只有当结果为 -1 (Strictly Worse) 时才由于违反 Pareto 拒绝
            if safe_compare(val_win, val_lose, p_dir) == -1:
                return False
        
        # B. Secondary Key Majority Voting
        # 规则：计算净胜票数，None 视为弃权
        votes = 0
        valid_comparisons = 0 # 记录有多少个指标是能够进行有效比较的
        total_constraints = 0
        
        for area_name, config in area_configs.items():
            constraints = config.get('secondary_constraints', {})
            total_constraints += len(constraints)
            if len(constraints) == 0:
                continue
            for metric, direction in constraints.items():
                val_win = get_metric_value(win_row, area_name, metric)
                val_lose = get_metric_value(lose_row, area_name, metric)
                
                result = safe_compare(val_win, val_lose, direction)
                votes += result
                
                if val_win is not None and val_lose is not None:
                    valid_comparisons += 1
        
        if total_constraints == 0:
            return True # 如果没有第二个次要众数投票，那么尊重primary metric的方向

        # 如果所有 Secondary Metrics 都是 None，我们该怎么办？
        # 这里的策略是：如果没有足够证据证明 Winner 更好，就保守地丢弃
        if valid_comparisons == 0:
            return False
        # 票数必须 > 0 (Winner 必须在次要指标上有总体优势)
        return votes > 0

    for pair in raw_pairs:
        if check_robust_filters(pair['chosen'], pair['rejected']):
            valid_pairs.append(pair)

    return valid_pairs

def format_transfer(group_pairs: List[dict], filt_out_types=None):
    training_pairs = []
    for pair in group_pairs:
        if (filt_out_types is not None) and (pair['type'] in filt_out_types):
            continue
        if random.random() > 0.5:
            training_pairs.append({
                "edited_image_paths": [
                    pair['chosen']['edited_image_path'],
                    pair['rejected']['edited_image_path']
                ],
                "instruction": pair['chosen']['instruction'],
                "source_image_path": pair['chosen']['source_image_path'],
                "gpt_response": GPT_RESPONSE.format(winner_value="\"Image A\"")
            })
        else:
            training_pairs.append({
                "edited_image_paths": [
                    pair['rejected']['edited_image_path'],
                    pair['chosen']['edited_image_path']
                ],
                "instruction": pair['chosen']['instruction'],
                "source_image_path": pair['chosen']['source_image_path'],
                "gpt_response": GPT_RESPONSE.format(winner_value="\"Image B\"")
            })
    return training_pairs

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, required=True, help='Editing Task Type')
    parser.add_argument('--prompts-num', type=int, default=1700, help='Number of prompts to generate for each task')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for the output files')
    parser.add_argument(
        '--thresholds-config-file', type=str,
        default="/data/open_edit/configs/pipelines/data_construction_configs.json",
        help='Path to the threshold config file for different tasks.'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default="/data/open_edit/data/c_annotated_group_data",
        help='Path to the input directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="/data/open_edit/data/d_train_data",
        help='Path to the output directory'
    )
    parser.add_argument(
        '--filt-out-strategy',
        type=str,
        choices=['head_tail', 'three_tiers'],
        required=True,
        help='Filtering strategy for training pair types'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks = args.tasks.split(',')
    prefix = args.prefix
    if prefix is None or prefix.strip() == "":
        data_save_path = args.output_dir
    else:
        prefix = prefix.lstrip("/\\")
        data_save_path = os.path.join(args.output_dir, prefix)
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    
    task_metrics_config_map = json.load(open(args.thresholds_config_file, 'r'))
    for task in tasks:
        input_task_json = os.path.join(args.input_dir, f"{task}_grouped.jsonl")
        area_configs = task_metrics_config_map.get(task, None)
        if area_configs is None:
            print(f"🚨 Config for processing [{task}] is missing!")
            continue
        raw_data = [
            json.loads(line) for line in open(input_task_json, 'r')
        ]

        cleaned_data = []
        for raw_group in raw_data:
            try:
                cleaned_group = [
                    item for item in raw_group['results']
                ]
                cleaned_data.append(cleaned_group)
            except Exception as e:
                print(f"Error processing group: {e}")

        all_training_pairs = []
        task_prompts_num = 0
        for group in tqdm(cleaned_data, desc=f"Processing task {task}", total=len(cleaned_data)):
            group_pairs = construct_pairs(group, area_configs)
            
            if args.filt_out_strategy == 'head_tail':
                filt_out_types = ['Elite_vs_Normal', 'Normal_vs_Failure']
            elif args.filt_out_strategy == 'three_tiers':
                filt_out_types = None
            else:
                filt_out_types = None # Fallback
            
            training_pairs = format_transfer(group_pairs, filt_out_types=filt_out_types)

            if len(training_pairs) < 1:
                continue
            all_training_pairs.extend(training_pairs)

            task_prompts_num += 1
            if task_prompts_num >= args.prompts_num:
                break
        print(f"Task {task}: Generated {len(all_training_pairs)} training pairs.")
        output_task_json = os.path.join(data_save_path, f"{task}.json")
        with open(output_task_json, 'w', encoding='utf-8') as f_out:
            json.dump(all_training_pairs, f_out, indent=4)
        print(f"✅ Successfully saved to {output_task_json}")
        

        
