import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils.logging_util import logger_init

import json
import pandas as pd
from tqdm import tqdm
from pvc_judge import PairWiseJudge
from prompts.prompt_manager import PromptAssetStore
from common_utils.dataset_loader import load_dataset
from core.cache_manager import CacheManager, generate_cache_key

CACHE_ROOT = "/mnt/jfs/lora_models_cache"

def infer_openedit_bench(judge, dataset, generation_config, cache_manager, output_file):
    item_to_process, all_results = dataset.load_cache(cache_manager)
    logger.info(f"Loaded {len(all_results)} cached results. {len(item_to_process)} items left to process.")
    if item_to_process:
        for item_key in tqdm(item_to_process, desc="Evaluating OpenEdit Bench"):
            item = dataset.get_item(item_key)
            winner, item_res_dict = judge(item, generation_config)
            if winner is None or winner.lower() not in ['image a', 'a', 'image_a', 'imagea', 'image b', 'b', 'image_b', 'imageb', 'tie', 'equal']:
                continue
            cache_manager.append(generate_cache_key(item_key), item_res_dict)
            all_results[item_key] = item_res_dict
    
    with open(output_file, 'w') as out_f:
        for item_key, item_res_dict in all_results.items():
            json_line = {
                "key": item_key,
                "winner": item_res_dict.get("winner", None),
                "result": item_res_dict,
            }
            out_f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

def infer_reward_bench(judge, dataset, generation_config, output_file):
    static_dict, all_results = {}, {}
    static_dict.setdefault("overall", {"correct": 0, "total": 0})
    for item_key in tqdm(dataset.dataset, desc="Evaluating Reward Bench"):
        item = dataset.get_item(item_key)
        gt_winner = item['winner']

        winner, item_res_dict = judge(item, generation_config)
        print(f"Item: {item_key}, GT Winner: {gt_winner}, Predicted Winner: {winner}")
        if winner is None or winner.lower() not in ['image a', 'a', 'image_a', 'imagea', 'image b', 'b', 'image_b', 'imageb', 'tie', 'equal']:
            continue
        if 'task_type' in item:
            static_dict.setdefault(item['task_type'], {"correct": 0, "total": 0})
            all_results.setdefault(item['task_type'], {})
            static_dict[item['task_type']]["total"] += 1
            static_dict["overall"]["total"] += 1
            if winner == gt_winner:
                static_dict[item['task_type']]["correct"] += 1
                static_dict["overall"]["correct"] += 1
            all_results[item['task_type']][item_key] = {
                "winner": winner,
                "gt_winner": gt_winner,
                "results": item_res_dict,
            }
        else:
            static_dict["overall"]["total"] += 1
            if winner == gt_winner:
                static_dict["overall"]["correct"] += 1
            all_results[item_key] = {
                "winner": winner,
                "gt_winner": gt_winner,
                "results": item_res_dict,
            }

    with open(output_file, 'w') as out_f:
        for task_type, task_results in all_results.items():
            for item_key, item_res_dict in task_results.items():
                json_line = {
                    "key": item_key,
                    "winner": item_res_dict.get("winner", None),
                    "gt_winner": item_res_dict.get("gt_winner", None),
                    "results": item_res_dict.get("results", None),
                    "task": task_type,
                }
                out_f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

    return static_dict

def report_board(res_dict, model_name, bench_name):
    print(f"\n=== Evaluation Results on {bench_name} ===")
    print(f"Model: {model_name}")
    print("=========================================")
    for task_type, static in res_dict.items():
        acc = static["correct"] / static["total"] if static["total"] > 0 else 0
        print(f"  {task_type}: {acc:.4f}")
    print("=========================================")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--use-flash-attn", action='store_true')
    parser.add_argument("--bmk", type=str, default="editscore")
    parser.add_argument("--bmk-config", type=str, default="/data/open_edit/configs/datasets/bmk.json")
    parser.add_argument("--output-dir", type=str, default="/data/open_edit/output/eval_res")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-min-pixels", type=int, default=256*28*28)
    parser.add_argument("--image-max-pixels", type=int, default=1024*28*28)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--num-pass", type=int, default=1, help="Majority voting passes for each pair.")
    parser.add_argument("--prompt-id", type=str, default=None, help="Specify the prompt id to use for evaluation.")
    parser.add_argument("--prompt-version", type=str, default=None, help="Specify the prompt version to use for evaluation.")
    parser.add_argument("--logger-level", type=str, default="INFO")
    parser.add_argument("--do-sample", action='store_true')
    parser.add_argument("--use-vllm", action='store_true')
    parser.add_argument("--save-details", action='store_true')
    parser.add_argument("--merged-model-cache-dir", type=str, default=CACHE_ROOT, help="Directory to cache merged models.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = logger_init(
        log_dir=f"{args.output_dir}/{pd.Timestamp.now().strftime('%m%d')}/{args.bmk}/logs",
        level=args.logger_level,
    )
    if args.use_vllm:
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    prompt_store = PromptAssetStore('assets')
    prompt_template = prompt_store.get_prompt(prompt_id=args.prompt_id, version=args.prompt_version)
    pvc_judge = PairWiseJudge(
        use_vllm=args.use_vllm,
        lora_model_path=args.lora_model_path,
        base_model_path=args.base_model_path,
        prompt_template=prompt_template,
        use_flash_attn=args.use_flash_attn,
        cache_root=args.merged_model_cache_dir,
    )
    if args.num_pass > 1:
        logger.info(f"Using majority voting with {args.num_pass} passes for each pair.")

    generation_config = {
        "max_retries": args.max_retries,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "seed": args.seed,
        "max_num_seqs": args.max_num_seqs,
        "num_pass": args.num_pass,
    }
    if not os.path.exists(args.bmk_config):
        raise FileNotFoundError(f"Reward benchmark config not found: {args.bmk_config}")
    with open(args.bmk_config, "r", encoding="utf-8") as f:
        bmk_config = json.load(f)
    bmk_info = dict(bmk_config[args.bmk])
    dataset = load_dataset(bmk_name=args.bmk, **bmk_info)
    output_file_dir = os.path.join(args.output_dir, pd.Timestamp.now().strftime('%m%d'), args.bmk)
    
    if args.bmk == "openedit":
        cache_dir = os.path.join(output_file_dir, ".cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        cache_manager = CacheManager(cache_file=cache_dir + 'gen.jsonl')
        infer_openedit_bench(
            judge=pvc_judge,
            dataset=dataset,
            generation_config=generation_config,
            cache_manager=cache_manager,
            output_file=os.path.join(output_file_dir, "results.jsonl"),
        )
    else:
        static_dict = infer_reward_bench(
            judge=pvc_judge,
            dataset=dataset,
            generation_config=generation_config,
            output_file=os.path.join(output_file_dir, "results.jsonl"),
        )
        report_board(static_dict, args.lora_model_path, args.bmk)