import io
import os
import json
import sys
from pathlib import Path
import megfile
import functools
import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor


def _bootstrap_import_path():
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    for p in (str(repo_root), str(src_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_import_path()

from core.cache_manager import CacheManager, generate_cache_key

TASK_EDIT_TYPE_MAP = {
    "season-transformation": "Apply seasonal transformation", # tone transfer
    "change-weather": "Change weather conditions", # tone transfer
    "adjust-global-lighting": "Adjust global lighting", # tone transfer
    "add-film-grain": "Add film grain or vintage filter", # tone transfer
    "change-color-tone": "Change overall color tone", # tone transfer
    "add-new-scene-context": "Add new scene context/background", # background change
    "strong-artistic-style-transfer": "Strong artistic style transfer", # style transfer
    "photo-to-cartoon": "Photo to cartoon/sketch/comic", # style transfer
    "modern-historical-style": "Modern \u2194 historical style/look", # style transfer
    "clothing-edit": "Clothing edit (change color/outfit)", # ps human
    "age-gender-edit": "Change age / gender", # ps human
    "2d-anime-style": "Convert person to 2D anime/manga style", # style transfer
    "pixar-style": "Convert person to Pixar/Disney-like", # style transfer
    "western-comic-style": "Convert person to Western comic cel-shaded style", # style transfer
    "line-art-style": "Line-art ink sketch of the person", # style transfer
    "stickerify-style": "Sticker-ify the person with bold outline and white border", # style transfer
    "caricature-style": "Caricature with mild feature exaggeration", # style transfer
    "funko-pop-style": "Funko-Pop\u2013style toy figure", # style transfer
    "lego-style": "LEGO-minifigure rendition", # style transfer
    "simpsonize-style": "Simpsonize the person", # style transfer
    "relocate-object": "Relocate an object (change its position/spatial relation)", # relation change
    "change-object-size-shape-orien": "Change the size/shape/orientation of an object", # relation change
    "pose-change": "Pose tweak (minor plausible change)", # motion change
    "expression-change": "Modify expressions (smile, frown, neutral)",
    "subject-add": "Add a new object to the scene", # local editing
    "subject-remove": "Remove an existing object", # local editing
    "subject-replace": "Replace one object category with another", # local editing
    "text-replace": "Replace text in signs/posters/billboards", # text
    "text-add": "Add new (handwritten/printed/etc) text", # text
    "text-modify": "Change font style or color of visible text if there is text", # text
    "text-translation": "Translate written text into other languages", # text
    "color-material-alter": "Change an object's attribute (e.g., color/material)",
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="subject-add", help='Task name for the image editing task.')
    parser.add_argument(
        '--output-dir', type=str,
        default="/data/open_edit/data/a_raw_img_prompt_pair_data",
        help='Path to save the prepared pico data.'
    )
    parser.add_argument('--image-save-path', type=str, default=None, help='Path to save the prepared pico data.')
    parser.add_argument(
        '--path-to-pico-sft-jsonl', type=str,
        default="/data/open_edit/data/a_raw_img_prompt_pair_data/pico_sft.jsonl",
        help='Path to the original pico sft jsonl file.'
    )
    return parser.parse_args()

def _url_to_image(url):
    response = requests.get(url, stream=True)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image

def _url_to_filename(url):
    return '_'.join(url.split('.com/')[1].split('/'))

def _url_to_id(url):
    return '_'.join(url.split('.com/')[1].split('/')).split('.')[0]

def process_single_item(item_key, data, source_image_save_path):
    # print(f"Image url: {data[0]}")
    image_url = data[0]
    instruction = data[1] # short instruction
    detailed_instruction = data[2] # long instruction
    if image_url:
        try:
            img_name = _url_to_filename(image_url)
            img_file = os.path.join(source_image_save_path, img_name)
            if not megfile.smart_exists(img_file):
                img = _url_to_image(image_url)
                with megfile.smart_open(img_file, 'wb') as out_f:
                    img.save(out_f, format='JPEG')
            else:
                print(f'Image {img_file} already exists, skipping download.')
                return item_key, None
            return item_key, {
                "image_path": img_file,
                "instruction": instruction,
                "detailed_instruction": detailed_instruction,
            }
        except Exception as e:
            print(f"Failed to process image from {image_url}: {e}")
    return item_key, None

def _load_item(data: dict, task):
    if TASK_EDIT_TYPE_MAP[task] in data['edit_type']:
        item_key = _url_to_id(data['open_image_input_url'])
        image_url = data['open_image_input_url']
        instruction = data['summarized_text']
        detailed_instruction = data['text']
        return [(item_key, (image_url, instruction, detailed_instruction))]
    return {}

def load_task_dataset_multithreaded(full_dataset, task):
    max_workers = os.cpu_count() or 1

    print(f"Processing dataset (length: {len(full_dataset)}) with {max_workers} threads", flush=True)
    items = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        partial_load_item = functools.partial(_load_item, task=task)
        results_iterator = tqdm(
            executor.map(partial_load_item, full_dataset),
            total=len(full_dataset),
            desc="Processing dataset with multiple threads"
        )
        
        for item_result in results_iterator:
            items.update(item_result)

    return items

if __name__ == "__main__":
    args = parse_args()
    task = args.task.replace('-', '_')
    cache_path = os.path.join(args.output_dir, '.cache')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f'{task}_prepare_pico.jsonl')
    cache_manager = CacheManager(cache_file)

    source_image_save_path = os.path.join(args.image_save_path, task, 'inputs')

    full_dataset = [
        json.loads(line) for line in open(args.path_to_pico_sft_jsonl, 'r')
    ]
    task_dataset = load_task_dataset_multithreaded(full_dataset, task)
    item_to_process = [
        item_key
        for item_key in task_dataset.keys()
        if cache_manager.get(generate_cache_key(item_key)) is None
    ]
    # load cached items
    all_scores = {}
    for item_key in task_dataset.keys():
        if item_key not in item_to_process:
            all_scores[item_key] = cache_manager.get(generate_cache_key(item_key))

    if item_to_process:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_single_item, item_key, task_dataset[item_key], source_image_save_path)
                for item_key in item_to_process
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                unit="item",
                desc="Processing",
            ):
                item_key, result = future.result()
                if result:
                    all_scores[item_key] = result
                    cache_manager.append(generate_cache_key(item_key), result)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'{task}.jsonl'), 'w') as json_f:
        for item in all_scores:
            json_line = {
                "key": item,
                "image_path": all_scores[item]['image_path'],
                "instruction": all_scores[item]['instruction'],
            }
            json_f.write(json.dumps(json_line) + '\n')
