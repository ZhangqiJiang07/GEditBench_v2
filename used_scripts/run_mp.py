import io
import os
import json
import torch
import torch.multiprocessing as mp
import megfile
from typing import List
from PIL import Image

from core.cache_manager import CacheManager, generate_cache_key
from autogen.constants import MODEL_PATH_MAP

def load_pipeline(model_name: str):
    if model_name in ["qwen-image-edit"]:
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
    raise ValueError(f"Unsupported model name: {model_name}")
    
def prepare_inputs(model_name, image, instruction):
    if model_name == "qwen-image-edit":
        return {
            "image": image,
            "prompt": instruction,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 28,
        }
    elif model_name in ["qwen-image-edit-2509", "qwen-image-edit-2511"]:
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
    elif model_name == "step1x_edit1.2":
        return {
            "image": image,
            "prompt": instruction,
            "num_inference_steps": 28,
            "true_cfg_scale": 6.0,
            "generator": torch.Generator(device="cuda").manual_seed(42),
            "enable_thinking_mode": False,
            "enable_reflection_mode": False,
        }
    elif model_name == "step1x_edit1.2-preview":
        return {
            "image": image,
            "prompt": instruction,
            "num_inference_steps": 28,
            "true_cfg_scale": 4.0,
            "generator": torch.Generator(device="cuda").manual_seed(42),
            "enable_thinking_mode": False,
            "enable_reflection_mode": False,
        }
    elif model_name == "kontext":
        return {
            "image": image,
            "prompt": instruction,
            "num_inference_steps": 28,
            "guidance_scale": 2.5,
            "generator": torch.Generator(device="cuda").manual_seed(42),
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
        # pipeline.set_progress_bar_config(disable=True)
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
                f"{job['key']}.webp",
            )

            inputs = prepare_inputs(model_name, image, job["instruction"])

            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]
                with megfile.smart_open(save_path, "wb") as out_f:
                    output_image.save(out_f, format="WEBP")

            result_queue.put({
                'key': job["key"],
                'image_path': save_path,
                'instruction': job["instruction"],
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
    # original_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    for rank, group in enumerate(gpu_groups):
        # try:
            # os.environ["CUDA_VISIBLE_DEVICES"] = group
        p = mp.Process(
            target=worker_loop,
            args=(rank, group, model_name, save_root, job_queue, result_queue),
        )
        p.start()
        workers.append(p)
        # finally:
        #     if original_env is None:
        #         os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        #     else:
        #         os.environ["CUDA_VISIBLE_DEVICES"] = original_env
    return workers



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="subject-add", help='Task name for the image editing task.')
    parser.add_argument('--model', type=str, default="qwen-image-edit", help='Model name to load the pipeline from.')
    parser.add_argument(
        '--dataset-path', type=str,
        default="/data/open_edit/data/b_filtered_img_prompt_pair_data",
        help='Dataset path for the image editing task.'
    )
    parser.add_argument('--gpus-per-worker', type=int, default=1, help='Number of GPUs to allocate per worker process.')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.gpus_per_worker < 1:
        raise ValueError("`--gpus-per-worker` 必须为正整数。")
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPU detected.")

    gpu_groups = build_gpu_groups(gpu_count, args.gpus_per_worker)
    if not gpu_groups:
        raise RuntimeError(
            f"无法按照每个 worker {args.gpus_per_worker} 张 GPU 的配置划分资源，"
            f"请确认当前机器可见 GPU 数为 {gpu_count}。"
        )
    print(f"Detected {gpu_count} GPUs, forming {len(gpu_groups)} GPU groups for workers: {gpu_groups}")

    cache_path = os.path.join(f'/data/{args.dataset_path}', args.task, 'gen_cache')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f'{args.model}.jsonl')
    cache_manager = CacheManager(cache_file)
    meta_info_path = os.path.join(f'/data/{args.dataset_path}', args.task, 'meta_info.jsonl')
    dataset = [
        json.loads(line) for line in open(meta_info_path, 'r')
    ]
    print(f"数据集中共有 {len(dataset)} 条数据。")
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

    output_image_save_path = os.path.join(
        's3://jiangzhangqi',
        '/'.join(args.dataset_path.split('/')[1:]),
        args.task, args.model
    )

    workers = launch_workers(
        gpu_groups,
        args.model,
        output_image_save_path,
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

    
    cache_manager = CacheManager(cache_file)
    print("写入结果文件...")
    gen_save_path = os.path.join(f"/data/{args.dataset_path}", args.task)
    os.makedirs(gen_save_path, exist_ok=True)
    with open(os.path.join(gen_save_path, f'{args.model}_generation_results.jsonl'), 'w') as json_f:
        for item in dataset:
            cached = cache_manager.get(generate_cache_key(item["key"]))
            if cached is None:
                print(f"🚨 Warning: no cache found for key {item['key']}, skipping.")
                continue
            json_line = {
                "key": cached["key"],
                "image_path": cached['image_path'],
                "instruction": cached['instruction'],
            }
            json_f.write(json.dumps(json_line) + '\n')

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mp.set_start_method("spawn", force=True)
    main()
