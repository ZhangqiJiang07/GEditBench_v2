import os
import json
import sys
from pathlib import Path
import megfile
from PIL import Image
from tqdm import tqdm


def _bootstrap_import_path():
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    for p in (str(repo_root), str(src_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_import_path()

from core.cache_manager import CacheManager, generate_cache_key

TASK_MAP = {
    'background_change': 'background',
    'motion_change': 'action',
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="background_change", help='Task name for the image editing task.')
    parser.add_argument(
        '--path-to-nano-data', type=str,
        default="/path/to/nano/consistency/data",
        help='Path to save the prepared nano consistent data.'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default="/data/open_edit/data/a_raw_img_prompt_pair_data",
        help='Path to save the raw nano consistent data.'
    )
    parser.add_argument(
        '--image-save-path', type=str,
        default=None, help='Path to RESAVE the prepared nano consistent data.'
    )
    parser.add_argument('--sample-num', type=int, default=4000, help='Number of samples to prepare for nano consistent data.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task = args.task.replace('-', '_')
    cache_path = os.path.join(args.output_dir, '.cache')
    os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f'{task}_prepare_nano.jsonl')
    cache_manager = CacheManager(cache_file)

    info_data = [
        json.loads(line) for line in megfile.smart_open(
            f"{args.path_to_nano_data}/json/{TASK_MAP[task]}_cleaned.jsonl", 'r'
        )
    ]
    
    short_data = {}
    for i, item_data in tqdm(enumerate(info_data), total=len(info_data)):
        if len(short_data) >= args.sample_num:
            break
        if not item_data['instruction']:
            continue

        if len(item_data['instruction']) < 300 and len(item_data['instruction']) > 10:
            item_key = f"nano_consistent_150k_{i:08d}"
            if cache_manager.get(generate_cache_key(item_key)) is not None:
                short_data[item_key] = cache_manager.get(generate_cache_key(item_key))
                continue
            
            input_image_path = os.path.join(
                args.path_to_nano_data,
                item_data['input_images'][0].replace("Nano-150k/", "")
            )
            if args.image_save_path:
                input_img_ext = input_image_path.split('.')[-1].lower()
                img_format = 'JPEG' if input_img_ext in ['jpg', 'jpeg'] else input_img_ext.upper()
                input_image_path = os.path.join(args.image_save_path, task, 'inputs', f'{item_key}.{input_img_ext}')
                input_img = Image.open(input_image_path).convert("RGB")
                with megfile.smart_open(input_image_path, 'wb') as out_f:
                    input_img.save(out_f, format=img_format)

            short_data[item_key] = {
                'image_path': input_image_path,
                'instruction': item_data['instruction'],
            }
            cache_manager.append(generate_cache_key(item_key), short_data[item_key])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'{task}.jsonl'), 'w') as json_f:
        for item_key, item in short_data.items():
            item = {
                'key': f"{item_key}",
                'image_path': item['image_path'],
                'instruction': item['instruction'],
            }
            json_f.write(json.dumps(item) + '\n')
