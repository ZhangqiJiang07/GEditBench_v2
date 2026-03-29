import json
import os
from typing import List, Optional, Tuple

import megfile
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

from autogen.constants import MODEL_PATH_MAP, TURBO_SIGMAS
from common_utils.project_paths import CONFIGS_ROOT, DATA_ROOT, PROJECT_ROOT, normalize_benchmark_name, resolve_project_path
from core.cache_manager import CacheManager, generate_cache_key


DATASET_MODELS = {
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "qwen-image-edit-2511",
    "step1x_edit1.2",
    "step1x_edit1.2-preview",
    "kontext",
}

GEDITV2_MODELS = {
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "qwen-image-edit-2511",
    "step1x_edit1.2",
    "step1x_edit1.2-preview",
    "kontext",
    "flux.2_dev",
    "flux.2_klein_9b",
    "flux.2_klein_4b",
    "longcat_image_edit",
    "glm_image",
    "flux.2_dev_turbo",
    "FireRed-Image-Edit-1.1",
}


def _resolve_dataset_root(dataset_path: str) -> str:
    return str(resolve_project_path(dataset_path))


def _resolve_output_root(output_root: str) -> str:
    resolved = resolve_project_path(output_root)
    return str(resolved)


def generate_suitable_shape(
    width: int,
    height: int,
    base_size: int,
    step_size: int = 32,
    range_scale: float = 0.4,
) -> Tuple[int, int]:
    size_min = int(np.floor(np.sqrt(base_size * base_size * range_scale) / step_size) * step_size)
    size_all = list(range(size_min, base_size, step_size))
    area = base_size * base_size
    aspect_size = []
    for size in size_all:
        ceil_size = int(np.ceil(area / size / step_size) * step_size)
        floor_size = int(np.floor(area / size / step_size) * step_size)
        aspect_size.append((size, ceil_size))
        if ceil_size != floor_size:
            aspect_size.append((size, floor_size))

    aspect_size.append((base_size, base_size))
    for h, w in aspect_size[::-1]:
        if h != w:
            aspect_size.append((w, h))

    suitable_shapes = np.array(aspect_size).tolist()
    t_h, t_w = suitable_shapes[0]
    min_error = abs(t_w / t_h - width / height)
    for h, w in suitable_shapes[1:]:
        error = abs(width / height - w / h)
        if error < min_error:
            min_error = error
            t_w, t_h = w, h
    return t_w, t_h


def process_image(img: Image.Image, img_size: int = 1024):
    w, h = img.size
    new_width, new_height = generate_suitable_shape(w, h, img_size)
    resized = img.resize((new_width, new_height), Image.LANCZOS)
    return resized, new_width, new_height


def load_pipeline(model_name: str):
    if model_name == "qwen-image-edit":
        from diffusers import QwenImageEditPipeline

        return QwenImageEditPipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    if model_name in ["qwen-image-edit-2509", "qwen-image-edit-2511"]:
        from diffusers import QwenImageEditPlusPipeline

        return QwenImageEditPlusPipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    if model_name in ["step1x_edit1.2", "step1x_edit1.2-preview"]:
        from diffusers import Step1XEditPipelineV1P2

        return Step1XEditPipelineV1P2.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    if model_name == "kontext":
        from diffusers import FluxKontextPipeline

        return FluxKontextPipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    if model_name == "flux.2_dev":
        from diffusers import Flux2Pipeline

        return Flux2Pipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    if model_name in ["flux.2_klein_9b", "flux.2_klein_4b"]:
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    if model_name == "longcat_image_edit":
        from diffusers import LongCatImageEditPipeline

        return LongCatImageEditPipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    if model_name == "glm_image":
        from diffusers.pipelines.glm_image import GlmImagePipeline

        return GlmImagePipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    if model_name == "flux.2_dev_turbo":
        from diffusers import Flux2Pipeline

        pipe = Flux2Pipeline.from_pretrained(
            MODEL_PATH_MAP["flux.2_dev"],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        pipe.load_lora_weights(
            MODEL_PATH_MAP[model_name],
            weight_name="flux.2-turbo-lora.safetensors",
        )
        return pipe
    if model_name == "FireRed-Image-Edit-1.1":
        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=None)
        return pipe
    raise ValueError(f"Unsupported model name: {model_name}")


def prepare_inputs(model_name: str, image: Image.Image, instruction: str, generation_mode: str):
    if generation_mode == "dataset":
        if model_name == "qwen-image-edit":
            return {
                "image": image,
                "prompt": instruction,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 28,
            }
        if model_name in ["qwen-image-edit-2509", "qwen-image-edit-2511"]:
            return {
                "image": [image],
                "prompt": instruction,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 28,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }
        if model_name == "step1x_edit1.2":
            return {
                "image": image,
                "prompt": instruction,
                "num_inference_steps": 28,
                "true_cfg_scale": 6.0,
                "generator": torch.Generator(device="cuda").manual_seed(42),
                "enable_thinking_mode": False,
                "enable_reflection_mode": False,
            }
        if model_name == "step1x_edit1.2-preview":
            return {
                "image": image,
                "prompt": instruction,
                "num_inference_steps": 28,
                "true_cfg_scale": 4.0,
                "generator": torch.Generator(device="cuda").manual_seed(42),
                "enable_thinking_mode": False,
                "enable_reflection_mode": False,
            }
        if model_name == "kontext":
            return {
                "image": image,
                "prompt": instruction,
                "num_inference_steps": 28,
                "guidance_scale": 2.5,
                "generator": torch.Generator(device="cuda").manual_seed(42),
            }
        raise ValueError(f"Unsupported model `{model_name}` for dataset generation mode")

    if generation_mode == "geditv2":
        if model_name == "qwen-image-edit":
            return {
                "image": image,
                "prompt": instruction,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50,
            }
        if model_name in ["qwen-image-edit-2509", "qwen-image-edit-2511"]:
            return {
                "image": [image],
                "prompt": instruction,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }
        if model_name == "step1x_edit1.2":
            return {
                "image": image,
                "prompt": instruction,
                "num_inference_steps": 50,
                "true_cfg_scale": 6.0,
                "generator": torch.Generator(device="cuda").manual_seed(42),
                "enable_thinking_mode": False,
                "enable_reflection_mode": False,
            }
        if model_name == "step1x_edit1.2-preview":
            return {
                "image": image,
                "prompt": instruction,
                "num_inference_steps": 50,
                "true_cfg_scale": 4.0,
                "generator": torch.Generator(device="cuda").manual_seed(42),
                "enable_thinking_mode": False,
                "enable_reflection_mode": False,
            }
        if model_name == "kontext":
            return {
                "image": image,
                "prompt": instruction,
                "num_inference_steps": 50,
                "guidance_scale": 2.5,
                "generator": torch.Generator(device="cuda").manual_seed(42),
            }
        if model_name == "flux.2_dev":
            return {
                "image": image,
                "prompt": instruction,
                "num_inference_steps": 50,
                "guidance_scale": 4,
                "generator": torch.Generator(device="cuda").manual_seed(42),
            }
        if model_name in ["flux.2_klein_9b", "flux.2_klein_4b"]:
            return {
                "image": image,
                "prompt": instruction,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "generator": torch.Generator(device="cuda").manual_seed(0),
            }
        if model_name == "longcat_image_edit":
            return {
                "image": [image],
                "prompt": instruction,
                "negative_prompt": "",
                "num_inference_steps": 50,
                "guidance_scale": 4.5,
                "num_images_per_prompt": 1,
                "generator": torch.Generator(device="cuda").manual_seed(43),
            }
        if model_name == "glm_image":
            resize_image, new_w, new_h = process_image(image, img_size=1024)
            return {
                "image": [resize_image],
                "prompt": instruction,
                "num_inference_steps": 50,
                "guidance_scale": 1.5,
                "height": new_h,
                "width": new_w,
                "generator": torch.Generator(device="cuda").manual_seed(42),
            }
        if model_name == "flux.2_dev_turbo":
            return {
                "image": image,
                "prompt": instruction,
                "sigmas": TURBO_SIGMAS,
                "num_inference_steps": 8,
                "guidance_scale": 2.5,
                "generator": torch.Generator(device="cuda").manual_seed(42),
            }
        if model_name == "FireRed-Image-Edit-1.1":
            return {
                "image": [image],
                "prompt": instruction,
                "generator": torch.Generator(device="cuda").manual_seed(49),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50,
                "num_images_per_prompt": 1,
            }
        raise ValueError(f"Unsupported model `{model_name}` for GEditBench v2 generation mode")

    raise ValueError(f"Unknown generation mode: {generation_mode}")


def worker_loop(
    rank: int,
    gpu_group: str,
    model_name: str,
    save_root: str,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    generation_mode: str,
    output_format: str,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
    print(f"[Worker {rank}] starting on GPUs {gpu_group}")

    try:
        pipeline = load_pipeline(model_name)
        print(f"[Worker {rank}] model loaded.")
    except Exception as exc:
        print(f"[Worker {rank}] model load failed: {exc}")
        result_queue.put({"worker_fail": rank})
        result_queue.put({"worker_done": rank})
        return

    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            if job["image_path"].startswith("s3://"):
                reader = megfile.smart_open(job["image_path"], "rb")
            else:
                reader = job["image_path"]

            image = Image.open(reader).convert("RGB")

            if generation_mode == "dataset":
                save_path = os.path.join(save_root, f"{job['key']}.webp")
            elif generation_mode == "geditv2":
                save_path = os.path.join(save_root, model_name, f"{job['key']}.png")
            else:
                raise ValueError(f"Unknown generation mode: {generation_mode}")

            if not save_path.startswith("s3://"):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            inputs = prepare_inputs(model_name, image, job["instruction"], generation_mode=generation_mode)
            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]
                with megfile.smart_open(save_path, "wb") as out_f:
                    output_image.save(out_f, format=output_format)

            result_queue.put(
                {
                    "key": job["key"],
                    "image_path": save_path,
                    "instruction": job["instruction"],
                }
            )
        except Exception as exc:
            print(f"[Worker {rank}] ERROR on {job.get('key')}: {exc}")
            continue

    result_queue.put({"worker_done": rank})
    print(f"[Worker {rank}] exit.")


def writer_loop(result_queue: mp.Queue, cache_file: str, worker_count: int):
    cache_manager = CacheManager(cache_file)
    finished_workers = 0
    while True:
        item = result_queue.get()
        if "worker_done" in item:
            finished_workers += 1
            if finished_workers == worker_count:
                break
            continue
        if "worker_fail" in item:
            print("[Writer] worker failed:", item["worker_fail"])
            continue
        cache_manager.append(generate_cache_key(item["key"]), item)
    print("[Writer] all workers finished.")


def build_gpu_groups(total_gpus: int, gpus_per_worker: int) -> List[str]:
    groups = []
    for start in range(0, total_gpus, gpus_per_worker):
        group = list(range(start, min(start + gpus_per_worker, total_gpus)))
        if len(group) < gpus_per_worker:
            break
        groups.append(",".join(map(str, group)))
    return groups


def launch_workers(
    gpu_groups: List[str],
    model_name: str,
    save_root: str,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    generation_mode: str,
    output_format: str,
):
    workers = []
    for rank, group in enumerate(gpu_groups):
        p = mp.Process(
            target=worker_loop,
            args=(rank, group, model_name, save_root, job_queue, result_queue, generation_mode, output_format),
        )
        p.start()
        workers.append(p)
    return workers


def _load_jsonl(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_dataset_generation(
    task: str,
    model: str,
    dataset_path: str,
    gpus_per_worker: int = 1,
    output_bucket_prefix: str = "data/generated_images",
) -> str:
    if model not in DATASET_MODELS:
        raise ValueError(f"Unsupported model for dataset generation: {model}")
    if gpus_per_worker < 1:
        raise ValueError("`gpus_per_worker` must be a positive integer")

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPU detected.")
    gpu_groups = build_gpu_groups(gpu_count, gpus_per_worker)
    if not gpu_groups:
        raise RuntimeError(
            f"Cannot build GPU groups with {gpus_per_worker} GPUs per worker from {gpu_count} visible GPUs."
        )
    print(f"Detected {gpu_count} GPUs, forming {len(gpu_groups)} worker groups: {gpu_groups}")

    task = task.replace("-", "_")
    dataset_root = _resolve_dataset_root(dataset_path)

    cache_path = os.path.join(dataset_root, task, "gen_cache")
    os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f"{model}.jsonl")
    cache_manager = CacheManager(cache_file)

    meta_info_path = os.path.join(dataset_root, task, "meta_info.jsonl")
    dataset = _load_jsonl(meta_info_path)
    print(f"Loaded {len(dataset)} samples.")
    pending = [item for item in dataset if cache_manager.get(generate_cache_key(item["key"])) is None]
    print(f"Pending jobs: {len(pending)}")
    if not pending:
        return os.path.join(dataset_root, task, f"{model}_generation_results.jsonl")

    job_queue = mp.Queue(maxsize=256)
    result_queue = mp.Queue(maxsize=256)

    writer = mp.Process(target=writer_loop, args=(result_queue, cache_file, len(gpu_groups)))
    writer.start()

    output_root = _resolve_output_root(output_bucket_prefix)
    output_image_save_path = os.path.join(output_root, task, model)
    workers = launch_workers(
        gpu_groups=gpu_groups,
        model_name=model,
        save_root=output_image_save_path,
        job_queue=job_queue,
        result_queue=result_queue,
        generation_mode="dataset",
        output_format="WEBP",
    )

    for job in pending:
        job_queue.put(job)
    print("All jobs dispatched.")

    for _ in workers:
        job_queue.put(None)
    for p in workers:
        p.join()
    writer.join()

    cache_manager = CacheManager(cache_file)
    output_file = os.path.join(dataset_root, task, f"{model}_generation_results.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as json_f:
        for item in dataset:
            cached = cache_manager.get(generate_cache_key(item["key"]))
            if cached is None:
                print(f"Warning: no cache found for key {item['key']}, skipping.")
                continue
            json_f.write(
                json.dumps(
                    {
                        "key": cached["key"],
                        "image_path": cached["image_path"],
                        "instruction": cached["instruction"],
                    }
                )
                + "\n"
            )
    return output_file


def prepare_geditv2_inputs(bench_path: str) -> List[dict]:
    from datasets import load_from_disk
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    meta_info = load_from_disk(bench_path)

    def _load_item(item):
        return {
            "key": item["key"],
            "instruction": item["instruction"],
            "image_path": item["source_image"], # Image.Image
        }
    items = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for item in tqdm(
            executor.map(_load_item, meta_info),
            total=len(meta_info),
            desc="Loading images",
        ):
            items.append(item)

    print(f"Loaded {len(items)} image-instruction samples!")
    return items


def _resolve_geditv2_bench_path(bench_path: Optional[str], bmk_config_path: str) -> str:
    if bench_path:
        return str(resolve_project_path(bench_path))
    with open(bmk_config_path, "r", encoding="utf-8") as f:
        bmk_config = json.load(f)
    return str(resolve_project_path(bmk_config["geditv2"]["bench_path"]))


def _merge_geditv2_to_metadata(
    metadata_file_path: str,
    model: str,
    dataset: List[dict],
    cache_manager: CacheManager,
) -> str:
    metadata_path = os.path.join(metadata_file_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = _load_jsonl(metadata_path)
    key_to_image = {
        item["key"]: cache_manager.get(generate_cache_key(item["key"]))["image_path"]
        for item in dataset
        if cache_manager.get(generate_cache_key(item["key"])) is not None
    }

    for item in metadata:
        if item["key"] in key_to_image:
            item["candidates"].append(
                {
                    "model": model,
                    "image": os.path.relpath(key_to_image[item["key"]], os.path.dirname(metadata_file_path)),
                }
            )

    import pandas as pd

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    merged_output_path = os.path.join(os.path.dirname(metadata_file_path), f"metadata_{timestamp}.jsonl")
    with open(merged_output_path, "w", encoding="utf-8") as out_f:
        for item in metadata:
            out_f.write(json.dumps(item) + "\n")
    return merged_output_path


def run_geditv2_generation(
    model: str,
    gpus_per_worker: int = 1,
    bench_path: Optional[str] = None,
    image_save_dir: str = str(PROJECT_ROOT / "geditv2_bench" / "images" / "edited"),
    merge_to_metadata: str = "path/to/candidates/gallery/metadata.jsonl",
    bmk_config_path: str = str(CONFIGS_ROOT / "datasets" / "bmk.json"),
) -> Tuple[str, Optional[str]]:
    if model not in GEDITV2_MODELS:
        raise ValueError(f"Unsupported model for GEditBench v2 generation: {model}")
    if gpus_per_worker < 1:
        raise ValueError("`gpus_per_worker` must be >= 1.")

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPU detected.")

    gpu_groups = build_gpu_groups(gpu_count, gpus_per_worker)
    if not gpu_groups:
        raise RuntimeError(
            f"Cannot build GPU groups with {gpus_per_worker} GPUs per worker from {gpu_count} visible GPUs."
        )
    print(f"Detected GPUs: {gpu_count}. Launching workers: {gpu_groups}")

    bmk_config_path = str(resolve_project_path(bmk_config_path))
    bench_path = _resolve_geditv2_bench_path(bench_path, bmk_config_path)
    image_save_dir = str(resolve_project_path(image_save_dir))
    dataset = prepare_geditv2_inputs(bench_path)

    cache_dir = os.path.join(image_save_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{model}.jsonl")
    cache_manager = CacheManager(cache_file)

    pending = [item for item in dataset if cache_manager.get(generate_cache_key(item["key"])) is None]
    print(f"Pending jobs: {len(pending)}")
    if not pending:
        output_file = os.path.join(image_save_dir, f"{model}_generation_results.jsonl")
        return output_file, None

    job_queue = mp.Queue(maxsize=256)
    result_queue = mp.Queue(maxsize=256)

    writer = mp.Process(target=writer_loop, args=(result_queue, cache_file, len(gpu_groups)))
    writer.start()

    workers = launch_workers(
        gpu_groups=gpu_groups,
        model_name=model,
        save_root=image_save_dir,
        job_queue=job_queue,
        result_queue=result_queue,
        generation_mode="geditv2",
        output_format="PNG",
    )

    for job in pending:
        job_queue.put(job)
    print("All jobs dispatched.")

    for _ in workers:
        job_queue.put(None)
    for p in workers:
        p.join()
    writer.join()

    cache_manager = CacheManager(cache_file)
    output_file = os.path.join(image_save_dir, f"{model}_generation_results.jsonl")
    with open(output_file, "w", encoding="utf-8") as writer_f:
        for item in dataset:
            cached = cache_manager.get(generate_cache_key(item["key"]))
            if cached is None:
                continue
            writer_f.write(
                json.dumps(
                    {
                        "key": cached["key"],
                        "image_path": cached["image_path"],
                        "instruction": cached["instruction"],
                    }
                )
                + "\n"
            )
    print("All jobs finished. Results saved to", output_file)

    merged_output = None
    if merge_to_metadata:
        merged_output = _merge_geditv2_to_metadata(
            bench_path=bench_path,
            model=model,
            dataset=dataset,
            cache_manager=cache_manager,
        )
        print("Merged generated image paths back to metadata. Saved to", merged_output)

    return output_file, merged_output


OPENEDIT_MODELS = GEDITV2_MODELS
prepare_openedit_inputs = prepare_geditv2_inputs
_resolve_openedit_bench_path = _resolve_geditv2_bench_path
_merge_openedit_to_metadata = _merge_geditv2_to_metadata


def run_openedit_generation(*args, **kwargs) -> Tuple[str, Optional[str]]:
    return run_geditv2_generation(*args, **kwargs)
