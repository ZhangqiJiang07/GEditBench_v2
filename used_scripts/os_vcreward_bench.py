'''
开源VCReward-Bench的脚本
'''
import os
import json
from pathlib import Path
from PIL import Image
import megfile

NEW_IMAGE_SAVE_DIR = "/data/open_edit/vcreward_bench"
VCREWARD_BENCH_CONFIG = {
    "image_data_path": "s3://sunzheng-public/gedit/geditv5/images",
    "json_file_path": "/data/gedit-rm-benchmark/geditv5/data",
    "tasks": [
        "background_change", "camera_motion", "color_alter", "cref", "enhancement", "extract_object",
        "material_alter", "motion_change", "ps_human", "relation_change", "size_change", "sketch2image",
        "style_transfer", "subject_add", "subject_remove", "subject_replace", "text_editing", "tone_transfer",
        "chart_edit", "in_image_text_translation", "sref"
    ]
}
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

def is_valid_path(image_path):
    ab_image_path = os.path.join(VCREWARD_BENCH_CONFIG["image_data_path"], image_path)
    if ab_image_path.startswith("s3://"):
        return megfile.smart_exists(ab_image_path)
    else:
        return os.path.exists(ab_image_path)

def get_image_ext(file_path):
    return Path(file_path).suffix.lower()[1:]

def download_and_convert_image(image_path: str, save_path: str):
    if image_path.startswith("s3://"):
        with megfile.smart_open(image_path, 'rb') as f:
            img = Image.open(f).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    img.save(save_path, format="PNG")

CD_MAP = {
    0: "A",
    1: "B",
}

def load_item(item: dict, task: str, idx):
    item_key = f"{task}_pair_{idx:06d}"
    source_image_path = os.path.join(VCREWARD_BENCH_CONFIG["image_data_path"], item["source_image_path"])
    # new_source_image_path = os.path.join(NEW_IMAGE_SAVE_DIR, "images", "source", f"{item_key}.png")
    # download_and_convert_image(source_image_path, new_source_image_path)
    # for i, img_path in enumerate(item['edited_image_paths']):
    #     download_and_convert_image(
    #         os.path.join(VCREWARD_BENCH_CONFIG["image_data_path"], img_path),
    #         os.path.join(NEW_IMAGE_SAVE_DIR, "images", "edited", f"image_{CD_MAP[i]}", f"{item_key}.png")
    #     )

    input_dict = {
        "key": item_key,
        "instruction": item["instruction"],
        "source_image": f"./images/source/{item_key}.png",
        "edited_images": [
            f"./images/edited/image_{CD_MAP[i]}/{item_key}.png" for i, img_path in enumerate(item['edited_image_paths'])
        ],
        "winner": f"Image {item['winner']}",
        "task": task,
    }
    return input_dict

if __name__ == "__main__":
    os.makedirs(f"{NEW_IMAGE_SAVE_DIR}/images/source", exist_ok=True)
    os.makedirs(f"{NEW_IMAGE_SAVE_DIR}/images/edited/image_A", exist_ok=True)
    os.makedirs(f"{NEW_IMAGE_SAVE_DIR}/images/edited/image_B", exist_ok=True)
    metadata = []
    for task in VCREWARD_BENCH_CONFIG["tasks"]:
        meta_info = [
            json.loads(line) for line in open(os.path.join(VCREWARD_BENCH_CONFIG["json_file_path"], f"{task}_pair.jsonl"), 'r')
        ]
        for idx, item in enumerate(meta_info):
            if not is_valid_path(item["source_image_path"]):
                print(f"Source Image path is not valid: {item['source_image_path']}")
                continue
            if not all(is_valid_path(path) for path in item['edited_image_paths']):
                print(f"Edited Image path is not valid: {item['edited_image_paths']}")
                continue
            item_info = load_item(item, TASK_NAME_MAP[task], idx)
            metadata.append(item_info)
        print(f"✅ Finished processing task {task}, total valid samples: {len(metadata)}")
    print(f"Total {len(metadata)} samples prepared for VCreward-Bench.")
    
    with open(os.path.join(NEW_IMAGE_SAVE_DIR, "metadata.jsonl"), 'w') as f:
        for jsonl_line in metadata:
            f.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")