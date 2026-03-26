import json
import os
from typing import List, Tuple
import megfile
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from megfile import SmartPath
from core.cache_manager import CacheManager, generate_cache_key
from autogen.constants import MODEL_PATH_MAP, TURBO_SIGMAS



def generate_suitable_shape(width: int, height: int, base_size: int, step_size: int = 32, range_scale: float = 0.4) -> Tuple[int, int]:
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
        pipeline = QwenImageEditPipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        return pipeline
    if model_name in ["qwen-image-edit-2509", "qwen-image-edit-2511"]:
        from diffusers import QwenImageEditPlusPipeline
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_PATH_MAP[model_name],
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        return pipeline
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
            torch_dtype=torch.bfloat16
        ).to("cuda")
        pipe.load_lora_weights(
            MODEL_PATH_MAP[model_name],
            weight_name="flux.2-turbo-lora.safetensors"
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


def prepare_inputs(model_name, image, instruction):
    if model_name == "qwen-image-edit":
        return {
            "image": image,
            "prompt": instruction,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }
    elif model_name in ["qwen-image-edit-2509", "qwen-image-edit-2511"]:
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
    elif model_name == "step1x_edit1.2":
        return {
            "image": image,
            "prompt": instruction,
            "num_inference_steps": 50,
            "true_cfg_scale": 6.0,
            "generator": torch.Generator(device="cuda").manual_seed(42),
            "enable_thinking_mode": False,
            "enable_reflection_mode": False,
        }
    elif model_name == "step1x_edit1.2-preview":
        return {
            "image": image,
            "prompt": instruction,
            "num_inference_steps": 50,
            "true_cfg_scale": 4.0,
            "generator": torch.Generator(device="cuda").manual_seed(42),
            "enable_thinking_mode": False,
            "enable_reflection_mode": False,
        }
    elif model_name == "kontext":
        return {
            "image": image,
            "prompt": instruction,
            "num_inference_steps": 50,
            "guidance_scale": 2.5,
            "generator": torch.Generator(device="cuda").manual_seed(42),
        }
    elif model_name == "flux.2_dev":
        return {
            "image": image,
            "prompt": instruction,
            "num_inference_steps": 50,
            "guidance_scale": 4,
            "generator": torch.Generator(device="cuda").manual_seed(42),
        }
    elif model_name in ["flux.2_klein_9b", "flux.2_klein_4b"]:
        return {
            "image": image,
            "prompt": instruction,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "generator": torch.Generator(device="cuda").manual_seed(0),
        }
    elif model_name == "longcat_image_edit":
        return {
            "image": [image],
            "prompt": instruction,
            "negative_prompt": '',
            "num_inference_steps": 50,
            "guidance_scale": 4.5,
            "num_images_per_prompt": 1,
            "generator": torch.Generator(device="cuda").manual_seed(43),
        }
    elif model_name == "glm_image":
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
    elif model_name == "flux.2_dev_turbo":
        return {
            "image": image,
            "prompt": instruction,
            "sigmas": TURBO_SIGMAS,
            "num_inference_steps": 8,
            "guidance_scale": 2.5,
            "generator": torch.Generator(device="cuda").manual_seed(42),
        }
    elif model_name == "FireRed-Image-Edit-1.1":
        return {
            "image": [image],
            "prompt": instruction,
            "generator": torch.Generator(device="cuda").manual_seed(49),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
            "num_images_per_prompt": 1,
        }
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def worker_loop(rank: int, gpu_group: str, model_name: str, save_root: str, job_queue: mp.Queue, result_queue: mp.Queue):
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
            save_path = os.path.join(
                save_root,
                model_name,
                f"{job['key']}.png",
            )

            inputs = prepare_inputs(model_name, image, job["instruction"])

            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]
                with megfile.smart_open(save_path, "wb") as out_f:
                    output_image.save(out_f, format="PNG")

            result_queue.put({
                "key": job["key"],
                "image_path": save_path,
                "instruction": job["instruction"],
            })
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


def launch_workers(gpu_groups: List[str], model_name: str, save_root: str, job_queue: mp.Queue, result_queue: mp.Queue):
    workers = []
    for rank, group in enumerate(gpu_groups):
        p = mp.Process(
            target=worker_loop,
            args=(rank, group, model_name, save_root, job_queue, result_queue),
        )
        p.start()
        workers.append(p)
    return workers


def prepare_geditv2_inputs(bench_path: str) -> list:
    meta_info = [
        json.loads(line) for line in open(os.path.join(bench_path, 'metadata.jsonl'), 'r')
    ]

    items = []
    for item in meta_info:
        items.append({
            "key": item['key'],
            "instruction": item["instruction"],
            "image_path": os.path.join(bench_path, item["source_image"])
        })
        
    print(f"Loaded {len(items)} image-instruction samples!")
    return items


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="GEditBench v2 multi-GPU inference runner.")
    parser.add_argument('--model', type=str, default="qwen-image-edit")
    parser.add_argument('--gpus-per-worker', type=int, default=1)
    parser.add_argument('--bench-path', type=str, default=None, help="Path to the GEditBench v2 benchmark directory containing metadata.jsonl and source images.")
    parser.add_argument('--image-save-dir', type=str, default='/path/to/geditv2/images/edited')
    parser.add_argument('--merge-to-metadata', action='store_true', help='Whether to merge generated image paths back to metadata.jsonl')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpus_per_worker < 1:
        raise ValueError("`--gpus-per-worker` must be >= 1.")

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPU detected.")

    gpu_groups = build_gpu_groups(gpu_count, args.gpus_per_worker)
    if not gpu_groups:
        raise RuntimeError(
            f"Cannot build GPU groups with {args.gpus_per_worker} GPUs per worker "
            f"from {gpu_count} visible GPUs."
        )
    print(f"Detected GPUs: {gpu_count}. Launching workers: {gpu_groups}")
    bench_path = args.bench_path
    if bench_path is None:
        with open("configs/datasets/bmk.json", 'r') as f:
            bmk_config = json.load(f)
        bench_path = bmk_config["geditv2"]["bench_path"]

    dataset = prepare_geditv2_inputs(bench_path)
    cache_dir = os.path.join(args.image_save_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{args.model}.jsonl")
    cache_manager = CacheManager(cache_file)

    pending = [
        item for item in dataset
        if cache_manager.get(generate_cache_key(item["key"])) is None
    ]
    print(f"Pending jobs: {len(pending)}")
    if not pending:
        return

    job_queue = mp.Queue(maxsize=256)
    result_queue = mp.Queue(maxsize=256)

    writer = mp.Process(
        target=writer_loop,
        args=(result_queue, cache_file, len(gpu_groups)),
    )
    writer.start()

    workers = launch_workers(
        gpu_groups,
        args.model,
        args.image_save_dir,
        job_queue,
        result_queue,
    )

    for job in pending:
        job_queue.put(job)
    print("All jobs dispatched.")

    for _ in workers:
        job_queue.put(None)

    for p in workers:
        p.join()

    writer.join()

    # Consolidate outputs
    cache_manager = CacheManager(cache_file)
    output_file = os.path.join(args.image_save_dir, f"{args.model}_generation_results.jsonl")
    with open(output_file, "w") as writer_f:
        for item in dataset:
            cached = cache_manager.get(generate_cache_key(item["key"]))
            if cached is None:
                continue
            writer_f.write(json.dumps({
                "key": cached["key"],
                "image_path": cached["image_path"],
                "instruction": cached["instruction"],
            }) + "\n")
    print("All jobs finished. Results saved to", output_file)

    if args.merge_to_metadata:
        metadata_path = os.path.join(bench_path, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found at {metadata_path}. Skipping merge.")
            return
        
        with open(metadata_path, "r") as meta_f:
            metadata = [json.loads(line) for line in meta_f]

        key_to_image = {
            item["key"]: cache_manager.get(generate_cache_key(item["key"]))["image_path"]
            for item in dataset
            if cache_manager.get(generate_cache_key(item["key"])) is not None
        }

        for item in metadata:
            if item["key"] in key_to_image:
                item["candidates"].append({
                    "model": args.model,
                    "image": os.path.relpath(key_to_image[item["key"]], bench_path)
                })
        import pandas as pd
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        merged_output_path = os.path.join(bench_path, f"metadata_{timestamp}.jsonl")
        with open(merged_output_path, "w") as out_f:
            for item in metadata:
                out_f.write(json.dumps(item) + "\n")
        print("Merged generated image paths back to metadata. Saved to", merged_output_path)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mp.set_start_method("spawn", force=True)
    main()
