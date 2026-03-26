'''
开源 GEditBench v2 的脚本
'''
import os
import json
from pathlib import Path
import shutil
from PIL import Image
import megfile
from typing import Dict
from megfile import SmartPath

GEDITV2_BENCH = "s3://jiangzhangqi/bench/geditv2"
NEW_IMAGE_SAVE_DIR = "data/geditv2_bench"
TASK_NAME_MAP = {
    "subject_add": "subject_addition",
    "subject_remove": "subject_removal",
    "subject_replace": "subject_replace",
    "size_adjustment": "size_adjustment",
    "color_alter": "color_alteration",
    "material_alter": "material_modification",
    "ps_human": "portrait_beautification",
    "motion_change": "motion_change",
    "relation_change": "relation_change",
    "text_editing": "text_editing",
    "background_change": "background_change",
    "style_change": "style_transfer",
    "tone_transfer": "tone_transfer",
    "camera_motion": "camera_motion",
    "line2image": "line2image",
    "character_reference": "character_reference",
    "object_reference": "object_reference",
    "style_reference": "style_reference",
    "chart_editing": "chart_editing",
    "in_image_text_translation": "in_image_text_translation",
    "enhancement": "enhancement",
    "hybrid": "hybrid",
    "openset": "openset"
}
MODEL_NAME_MAP = {
    'bagel': 'BAGEL',
    "flux2_dev": "FLUX2_dev",
    "flux2_dev_turbo": "FLUX2_dev_turbo",
    "flux2_klein_4b": "FLUX2_klein_4b",
    "flux2_klein_9b": "FLUX2_klein_9b",
    "gemini": "Nano_Banana_Pro",
    "glm_image": "GLM_Image",
    "kontext": "FLUX1_Kontext_dev",
    "longcat_image_edit": "LongCat_Image_Edit",
    "omnigen2": "OmniGen2",
    "gpt_image_1d5": "GPT_Image_1p5",
    "qwen_image_edit": "Qwen_Image_Edit",
    "qwen_image_edit_2509": "Qwen_Image_Edit_2509",
    "qwen_image_edit_2511": "Qwen_Image_Edit_2511",
    "seedream4d5": "Seedream4p5",
    "step1x_editv1p2": "Step1X_Edit_v1p2",
    "fire_red_image_edit_1p1": "FireRed_Image_Edit_1p1",
}

def load_model_candidates(model_lists: dict):
    candidates_dict = {}
    for model_name, model_results_jsonl_path in model_lists.items():
        candidates = {}
        for _line in open(model_results_jsonl_path, 'r'):
            line = json.loads(_line.strip())
            candidates[line['key']] = line['image_path']
        candidates_dict[model_name] = candidates
    return candidates_dict

def get_image_ext(file_path):
    return Path(file_path).suffix.lower()[1:]

def load_geditv2_bench(path: str):
    bench_path = SmartPath(path)
    subtask_folds = [child.name for child in bench_path.iterdir() if child.is_dir()]
    records = {}
    for fold in subtask_folds:
        task = '_'.join(fold.split('_')[:-1])
        new_task_name = TASK_NAME_MAP[task]
        meta = json.load(megfile.smart_open(f"{path}/{fold}/vis.json", "r"))
        for idx, (key, instruction) in enumerate(meta.items()):
            image_candidates = megfile.smart_glob(f"{path}/{fold}/images/{key}.*")
            image_path = image_candidates[0]
            image_ext = get_image_ext(image_path)
            if not image_candidates:
                continue
            records[key] = {
                "new_key": f"{new_task_name}_{idx:06d}",
                "instruction": instruction,
                "image_path": image_path,
                "task": new_task_name,
                "image_ext": image_ext
            }
    print(f"Loaded {len(records)} GEditBench v2 samples.")
    return records

def download_and_convert_image(image_path: str, save_path: str):
    if image_path.startswith("s3://"):
        with megfile.smart_open(image_path, 'rb') as f:
            img = Image.open(f).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    img.save(save_path, format="PNG")

def prepare_geditv2_bench_hf(
    meta_info: Dict,
    candidates_dicts: Dict,
) -> Dict[str, Dict]:
    metadata = []
    for item_key, item_meta_info in meta_info.items():
        existing_image_paths = {
            model_name: model_candidates_dict[item_key]
            for model_name, model_candidates_dict in candidates_dicts.items()
            if model_candidates_dict.get(item_key, None) is not None
        }
        source_image_dir = os.path.join(NEW_IMAGE_SAVE_DIR, "images", "source")
        os.makedirs(source_image_dir, exist_ok=True)
        new_source_image_path = os.path.join(source_image_dir, f"{item_meta_info['new_key']}.png")
        # download_and_convert_image(item_meta_info['image_path'], new_source_image_path)
        candidates = []
        for model_name, image_path in existing_image_paths.items():
            new_model_name = MODEL_NAME_MAP.get(model_name, model_name)
            new_edited_image_path = os.path.join(NEW_IMAGE_SAVE_DIR, "images", "edited", new_model_name, f"{item_meta_info['new_key']}.png")
            os.makedirs(os.path.dirname(new_edited_image_path), exist_ok=True)
            # download_and_convert_image(image_path, new_edited_image_path)
            candidates.append({
                "model": new_model_name,
                "image": f"./images/edited/{new_model_name}/{item_meta_info['new_key']}.png"
            })

        metadata.append({
            "key": item_meta_info['new_key'],
            "task": item_meta_info['task'],
            "instruction": item_meta_info['instruction'],
            "source_image": f"./images/source/{item_meta_info['new_key']}.png",
            "candidates": candidates,
        })
    return metadata


if __name__ == "__main__":
    with open("configs/datasets/geditv2_candidates.json", 'r') as f:
        candidate_models_config = json.load(f)
    
    meta_info = load_geditv2_bench(GEDITV2_BENCH)
    candidates_dicts = load_model_candidates(candidate_models_config)
    metadata = prepare_geditv2_bench_hf(meta_info, candidates_dicts)

    with open(os.path.join(NEW_IMAGE_SAVE_DIR, "metadata.jsonl"), 'w') as f:
        for jsonl_line in metadata:
            f.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
