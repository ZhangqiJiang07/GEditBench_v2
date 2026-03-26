import os
import json
import numpy as np

TASK_NAME_MAP = {
    "subject_add": "subject_addition",
    "subject_remove": "subject_removal",
    "subject_replace": "subject_replace",
    "size_change": "size_adjustment",
    "color_alter": "color_alteration",
    "material_alter": "material_modification",
    "ps_human": "portrait_beautification",
    "motion_change": "motion_change",
    "relation_change": "relation_change",
    "text_editing": "text_editing",
    "background_change": "background_change",
    "style_transfer": "style_transfer",
    "tone_transfer": "tone_transfer",
    "camera_motion": "camera_motion",
    "sketch2image": "line2image",
    "cref": "character_reference",
    "extract_object": "object_reference",
    "sref": "style_reference",
    "chart_edit": "chart_editing",
    "in_image_text_translation": "in_image_text_translation",
    "enhancement": "enhancement"
}

def calc_vc_reward(args):
    print("=" * 20 + " Calculating VC Reward " + "=" * 20)
    result_file = [
        json.loads(line) for line in open(args.file_path, 'r')
    ]
    all_task_static = {}
    for item in result_file:
        item_key = item["key"]
        task_type = item_key.split("_pair_")[0]
        all_task_static.setdefault(task_type, {"correct_num": 0, "total_num": 0})
        pair_result = item["results"]
        if pair_result["gt_winner"] == pair_result["winner"]:
            all_task_static[task_type]["correct_num"] += 1
        all_task_static[task_type]["total_num"] += 1

    overall_correct = sum([static["correct_num"] for static in all_task_static.values()])
    overall_total = sum([static["total_num"] for static in all_task_static.values()])
    
    for task_type, static in all_task_static.items():
        acc = static["correct_num"] / static["total_num"] if static["total_num"] > 0 else 0
        print(f"{task_type} >> ACC (%): {acc * 100:.2f} ({static['correct_num']}/{static['total_num']})")
    print("-" * 60)
    print(f"Overall Accuracy: {overall_correct / overall_total if overall_total > 0 else 0:.4f} ({overall_correct}/{overall_total})")
    print("=" * 60)
    mess = ''
    for task in [
        'subject_add', 'subject_remove', 'subject_replace', 'size_change', 'color_alter', 'material_alter', 'ps_human',
        'motion_change', 'relation_change', 'text_editing', 'in_image_text_translation', 'chart_edit', 'background_change',
        'style_transfer', 'tone_transfer', 'enhancement', 'camera_motion', 'sketch2image', 'cref', 'extract_object', 'sref'
    ]:
        mess += f"{all_task_static[TASK_NAME_MAP[task]]['correct_num']/all_task_static[TASK_NAME_MAP[task]]['total_num']*100:.2f}, " if task != 'sref' else f"{all_task_static[TASK_NAME_MAP[task]]['correct_num']/all_task_static[TASK_NAME_MAP[task]]['total_num']*100:.2f}"
    print(mess)
        
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("=" * 20 + " Basic Information " + "=" * 20)
    print(f"Pipeline: {args.file_path.split('/')[-3]}")
    print(f"Config: {args.file_path.split('/')[-2]}")
    calc_vc_reward(args)