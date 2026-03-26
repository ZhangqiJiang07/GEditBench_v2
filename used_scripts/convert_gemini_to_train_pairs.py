import os
import json
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from typing import List


GPT_RESPONSE = '''```json
{{
    "winner": {winner_value}
}}
```
'''

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, required=True, help='Editing Task Type')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--prompts-num', type=int, default=1700, help='Number of image-instruction pairs per training sample')
    parser.add_argument(
        '--input-dir', type=str,
        default="/data/open_edit/data/c_annotated_group_data",
        help='Directory containing the input JSONL files for each task.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="/data/open_edit/data/d_train_data",
        help='Path to the output directory'
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

    for task in tasks:
        input_task_json = os.path.join(args.input_dir, f"{task}_grouped.jsonl")
        
        raw_data = [
            json.loads(line) for line in open(input_task_json, 'r')
        ]

        all_training_pairs = []
        recorded_image_instruction_pairs = set()
        for item in tqdm(raw_data, desc=f"Processing task {task}"):
            try:
                item_res = item['results']
                # Record image-instruction pairs for data ablation
                image_prompt_key = item['key'].split('_pair_')[0]
                recorded_image_instruction_pairs.add(image_prompt_key)

                all_training_pairs.append({
                    "edited_image_paths": item_res['input_dict']['edited_images'],
                    "instruction": item_res['input_dict']['instruction'],
                    "source_image_path": item_res['input_dict']['source_image'],
                    "gpt_response": GPT_RESPONSE.format(winner_value=f"\"{item_res['winner']}\"")
                })
                
            except Exception as e:
                print(f"Error processing item: {e}")
            
            if len(recorded_image_instruction_pairs) >= args.prompts_num:
                break
        print(f"Total training pairs for task {task}: {len(all_training_pairs)}")
        output_task_json = os.path.join(data_save_path, f"{task}.json")
        with open(output_task_json, 'w', encoding='utf-8') as f_out:
            json.dump(all_training_pairs, f_out, indent=4)
        print(f"✅ Successfully saved to {output_task_json}")
