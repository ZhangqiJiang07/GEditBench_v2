import os
import io
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from autogen.utils.qwen3_vl_embedding import Qwen3VLEmbedder
from autogen.utils.kcenter_greedy import kCenterGreedy
from common_utils.image_util import open_image
from autogen.constants import QWEN3_VL_EMBEDDING_MODEL_PATH


def open_image_from_parquet(parquet_file: str, row_idx: int) -> Image.Image:
    if row_idx < 0:
        raise ValueError(f"row_idx must be >= 0, got {row_idx}")
    df = pd.read_parquet(parquet_file, columns=["src_image"])
    
    if row_idx >= len(df):
        raise IndexError(f"row_idx {row_idx} out of range for parquet with {len(df)} rows")
    image_bytes = df.iloc[row_idx]["src_image"]
    if image_bytes is None:
        raise ValueError(f"`src_image` is None at row_idx={row_idx}")
    
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    elif isinstance(image_bytes, bytearray):
        image_bytes = bytes(image_bytes)
    elif not isinstance(image_bytes, bytes):
        raise TypeError(
            f"`src_image` at row_idx={row_idx} is {type(image_bytes).__name__}, expected bytes-like"
        )

    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.copy()
    
def save_image_from_parquet(key: str, parquet_file: str, row_idx: int, save_dir: str) -> str:
    image = open_image_from_parquet(parquet_file, row_idx)
    save_path = os.path.join(save_dir, f"{key}.webp")
    try:
        import megfile
        with megfile.smart_open(save_path, 'wb') as out_f:
            image.save(out_f, format='WEBP')
    except ImportError:
        image.save(save_path, format='WEBP')
    return save_path

class MyQwen3VLEmbedder:
    def __init__(self, model_name_or_path="Qwen/Qwen3-VL-Embedding-8B"):
        self.model = Qwen3VLEmbedder(
            model_name_or_path=model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    def prepare_inputs(self, items):
        input_list = []
        for item in items:
            image = None
            if 'parquet_file' in item and 'row_idx' in item:
                try:
                    image = open_image_from_parquet(item['parquet_file'], item['row_idx'])
                except Exception as e:
                    print(f"Error opening image from parquet for key {item['key']}: {e}")
                    continue
            elif 'image_path' in item:
                try:
                    image = open_image(item['image_path'])
                except Exception as e:
                    print(f"Error opening image from path for key {item['key']}: {e}")
                    continue
            else:
                print(f"No valid image source for key {item['key']}, skipping.")
                continue
            input_list.append({
                "text": item['instruction'],
                "image": image
            })
        return input_list

    @torch.no_grad()
    def get_embeddings(self, meta_info, batch_size=4):
        all_embeddings = []
        for i in tqdm(range(0, len(meta_info), batch_size), desc="Qwen3-VL Encoding"):
            items = meta_info[i : i + batch_size]
            batch_inputs = self.prepare_inputs(items)
            batch_embeddings = self.model.process(batch_inputs)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu().to(torch.float32).numpy())
        return np.vstack(all_embeddings)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-num', type=int, default=1500)
    parser.add_argument('--task', type=str, default="background_change", help='Task name for the image editing task.')
    parser.add_argument(
        '--input-file', type=str,
        default="/data/open_edit/data/a_raw_img_prompt_pair_data/subject_add.jsonl"
    )
    parser.add_argument(
        '--output-dir', type=str,
        default="/data/open_edit/data/b_filtered_img_prompt_pair_data",
        help='Path to save the filtered data.'
    )
    parser.add_argument('--qwen-embedding-model-path', type=str, default=QWEN3_VL_EMBEDDING_MODEL_PATH)
    parser.add_argument(
        '--image-save-path', type=str,
        default=None, help='Path to save the input images of the filtered data.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    task = args.task.replace('-', '_')
    os.makedirs(os.path.join(args.output_dir, task), exist_ok=True)
    if 'UnicEdit-10M' in args.input_file:
        assert args.image_save_path is not None, "For parquet-based input, --image-save-path must be specified to save the extracted images."

    print("Loading Qwen3-VL-Embedding...")
    embedder = MyQwen3VLEmbedder(args.qwen_embedding_model_path)
    all_meta_info = [
        json.loads(line) for line in open(args.input_file, 'r')
    ]
    meta_info = []
    for item_info in tqdm(all_meta_info, total=len(all_meta_info), desc="Sample Filter..."):
        
        if 'UnicEdit-10M' in args.input_file:
            meta_info.append({
                "key": item_info['key'],
                "instruction": item_info['prompt_en'],
                "parquet_file": item_info['parquet_file'],
                "row_idx": item_info['row_idx']
            })
        else:
            meta_info.append({
                "key": item_info['key'],
                "instruction": item_info['instruction'],
                "image_path": item_info['image_path']
            })
    print(f"Final Meta Info Number: {len(meta_info)}")

    embeddings = embedder.get_embeddings(meta_info, batch_size=256)
    # print(f"[Embedding Shape]: {embeddings.shape}")
    sampler = kCenterGreedy(embeddings, metric='cosine')
    selected_indices = sampler.select_batch_(already_selected=[], N=args.sample_num)

    selected_items_info = [meta_info[i] for i in selected_indices]

    with open(os.path.join(args.output_dir, task, 'meta_info.jsonl'), 'wa') as json_f:
        for line in selected_items_info:
            if 'parquet_file' in line and 'row_idx' in line:
                image_path = save_image_from_parquet(
                    line['key'], line['parquet_file'], line['row_idx'], os.path.join(args.image_save_path, task, 'input')
                )
            else:
                image_path = line['image_path']
            json_line = {
                "key": line['key'],
                "instruction": line['instruction'],
                "image_path": line['image_path']
            }
            json_f.write(json.dumps(json_line) + '\n')
