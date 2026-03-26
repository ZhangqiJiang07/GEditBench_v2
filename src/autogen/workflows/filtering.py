import io
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from autogen.constants import QWEN3_VL_EMBEDDING_MODEL_PATH
from autogen.utils.kcenter_greedy import kCenterGreedy
from common_utils.image_util import open_image


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
    os.makedirs(save_dir, exist_ok=True)
    image = open_image_from_parquet(parquet_file, row_idx)
    save_path = os.path.join(save_dir, f"{key}.webp")
    try:
        import megfile

        with megfile.smart_open(save_path, "wb") as out_f:
            image.save(out_f, format="WEBP")
    except ImportError:
        image.save(save_path, format="WEBP")
    return save_path


class _Qwen3VLEmbedder:
    def __init__(self, model_name_or_path: str):
        from autogen.utils.qwen3_vl_embedding import Qwen3VLEmbedder

        self.model = Qwen3VLEmbedder(
            model_name_or_path=model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def prepare_inputs(self, items: List[dict]) -> Tuple[List[dict], List[dict]]:
        input_list = []
        valid_items = []
        for item in items:
            image = None
            if "parquet_file" in item and "row_idx" in item:
                try:
                    image = open_image_from_parquet(item["parquet_file"], item["row_idx"])
                except Exception as e:
                    print(f"Error opening image from parquet for key {item['key']}: {e}")
                    continue
            elif "image_path" in item:
                try:
                    image = open_image(item["image_path"])
                except Exception as e:
                    print(f"Error opening image from path for key {item['key']}: {e}")
                    continue
            else:
                print(f"No valid image source for key {item['key']}, skipping.")
                continue

            input_list.append({"text": item["instruction"], "image": image})
            valid_items.append(item)

        return input_list, valid_items

    @torch.no_grad()
    def get_embeddings(self, meta_info: List[dict], batch_size: int = 4):
        all_embeddings = []
        valid_meta_info = []
        for i in tqdm(range(0, len(meta_info), batch_size), desc="Qwen3-VL Encoding"):
            items = meta_info[i : i + batch_size]
            batch_inputs, batch_valid_items = self.prepare_inputs(items)
            if not batch_inputs:
                continue

            batch_embeddings = self.model.process(batch_inputs)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu().to(torch.float32).numpy())
            valid_meta_info.extend(batch_valid_items)

        if not all_embeddings:
            raise RuntimeError("No valid samples found for embedding.")

        return np.vstack(all_embeddings), valid_meta_info


def _build_meta_info(all_meta_info: List[dict], is_unicedit: bool) -> List[dict]:
    meta_info = []
    for item_info in tqdm(all_meta_info, total=len(all_meta_info), desc="Sample Filter..."):
        if is_unicedit:
            meta_info.append(
                {
                    "key": item_info["key"],
                    "instruction": item_info["prompt_en"],
                    "parquet_file": item_info["parquet_file"],
                    "row_idx": item_info["row_idx"],
                }
            )
        else:
            meta_info.append(
                {
                    "key": item_info["key"],
                    "instruction": item_info["instruction"],
                    "image_path": item_info["image_path"],
                }
            )
    return meta_info


def filter_dataset(
    sample_num: int,
    task: str,
    input_file: str,
    output_dir: str,
    qwen_embedding_model_path: str = QWEN3_VL_EMBEDDING_MODEL_PATH,
    image_save_path: str = None,
    embedding_batch_size: int = 256,
) -> str:
    task = task.replace("-", "_")
    is_unicedit = "UnicEdit-10M" in input_file

    task_output_dir = os.path.join(output_dir, task)
    os.makedirs(task_output_dir, exist_ok=True)

    if is_unicedit and image_save_path is None:
        raise ValueError("For parquet-based input, --image-save-path is required.")

    print("Loading Qwen3-VL-Embedding...")
    embedder = _Qwen3VLEmbedder(qwen_embedding_model_path)

    with open(input_file, "r", encoding="utf-8") as f:
        all_meta_info = [json.loads(line) for line in f]

    meta_info = _build_meta_info(all_meta_info, is_unicedit=is_unicedit)
    print(f"Final Meta Info Number: {len(meta_info)}")

    embeddings, valid_meta_info = embedder.get_embeddings(meta_info, batch_size=embedding_batch_size)
    sampler = kCenterGreedy(embeddings, metric="cosine")
    sample_num = min(sample_num, len(valid_meta_info))
    selected_indices = sampler.select_batch_(already_selected=[], N=sample_num)
    selected_items_info = [valid_meta_info[i] for i in selected_indices]

    output_file = os.path.join(task_output_dir, "meta_info.jsonl")
    with open(output_file, "w", encoding="utf-8") as json_f:
        for line in selected_items_info:
            if "parquet_file" in line and "row_idx" in line:
                image_path = save_image_from_parquet(
                    line["key"],
                    line["parquet_file"],
                    line["row_idx"],
                    os.path.join(image_save_path, task, "input"),
                )
            else:
                image_path = line["image_path"]

            json_line = {
                "key": line["key"],
                "instruction": line["instruction"],
                "image_path": image_path,
            }
            json_f.write(json.dumps(json_line) + "\n")

    return output_file
