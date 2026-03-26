import os
import json
import functools
import time
import megfile
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from datasets import Dataset
from abc import ABC, abstractmethod
from typing import Dict
from concurrent.futures import ProcessPoolExecutor
from core.cache_manager import generate_cache_key
from common_utils.pairwise import generate_canonical_pairs
from common_utils.logging_util import get_logger
from common_utils.project_paths import DEFAULT_BENCHMARK_NAME, normalize_benchmark_name, resolve_project_path
logger = get_logger()


class BaseDataset(ABC):
    @abstractmethod
    def prepare_dataset(self):
        pass
    
    def load_cache(self, cache_manager):
        item_to_process = []
        cached_res = {}
        for item_key in self.dataset.keys():
            if cache_manager.get(generate_cache_key(item_key)) is None:
                item_to_process.append(item_key)
            else:
                cached_res[item_key] = cache_manager.get(generate_cache_key(item_key))
        return item_to_process, cached_res
    
    def get_item(self, item_key):
        return self.dataset[item_key]

    def __len__(self):
        return len(self.dataset)


class EditScoreRewardBenchmark(BaseDataset):
    def __init__(self, data_path, eval_dim, shuffled_res_path, max_workers: int = None):
        self.data_path = str(resolve_project_path(data_path))
        self.eval_dim = eval_dim
        self.shuffled_res_path = str(resolve_project_path(shuffled_res_path))

        start_time = time.time()
        self.dataset = self.prepare_dataset(max_workers)
        print(f"EditScore dataset loaded in {time.time() - start_time} seconds")

    def _load_item(self, item: dict, shuffle_dict: dict = None):
        if item['dimension'] != self.eval_dim:
            return None
        pair_key = "_vs_".join(item['key'])
        item_key = "_task_".join([pair_key, item['task_type']])
        input_dict = {
            "instruction": item["instruction"],
            "input_image": item["input_image"],
            "edited_images": item["output_images"][::-1] if shuffle_dict[item_key] else item["output_images"],
            "winner": "Image B" if shuffle_dict[item_key] else "Image A",
            "task_type": item['task_type'],
        }
        return [(item_key, input_dict)]
    
    def prepare_dataset(self, max_workers)-> Dict[str, Dict]:
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        print(f"Preparing EditScore dataset using {max_workers} workers...")
        dataset = load_from_disk(self.data_path)
        shuffle_dict = json.load(open(self.shuffled_res_path, 'r'))
        
        items = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            partial_load_func = functools.partial(self._load_item, shuffle_dict=shuffle_dict)
            for item_info in tqdm(
                executor.map(partial_load_func, dataset),
                total=len(dataset),
                desc="Processing dataset with multiple processes"
            ):
                if item_info is None:
                    continue
                items.update(item_info)

        return items

class EditRewardRewardBenchmark(BaseDataset):
    def __init__(self, image_data_path, json_file_path):
        self.image_data_path = image_data_path if image_data_path.startswith("s3://") else str(resolve_project_path(image_data_path))
        self.json_file_path = str(resolve_project_path(json_file_path))

        start_time = time.time()
        self.dataset = self.prepare_dataset()
        print(f"EditReward dataset loaded in {time.time() - start_time} seconds")
    
    def _is_valid_path(self, image_path):
        if image_path.startswith("s3://"):
            return megfile.smart_exists(image_path)
        else:
            return os.path.exists(image_path)
    
    def _load_item(self, item):
        item_key = item['key']
        input_dict = {
            "instruction": item["instruction"],
            "input_image": item['source_image_path'],
            "edited_images": item['edited_image_paths'],
            "winner": item['winner'],
        }
        return [(item_key, input_dict)]
    
    def prepare_dataset(self) -> Dict[str, Dict]:
        items = {}
        meta_info = [
            json.loads(line) for line in open(self.json_file_path, 'r')
        ]
        for item in meta_info:
            if not self._is_valid_path(item["source_image_path"]):
                print(f"Source Image path is not valid: {item['source_image_path']}")
                continue
            if not all(self._is_valid_path(path) for path in item['edited_image_paths']):
                print(f"Edited Image path is not valid: {item['edited_image_paths']}")
                continue
            item_info = self._load_item(item)
            items.update(item_info)
        return items


class VCRewardBenchmark(BaseDataset):
    def __init__(self, bench_path):
        self.bench_path = str(resolve_project_path(bench_path))

        start_time = time.time()
        self.dataset = self.prepare_dataset()
        print(f"VCReward dataset loaded in {time.time() - start_time} seconds: {len(self.dataset)} valid samples found.")
    
    def prepare_dataset(self) -> Dict[str, Dict]:
        meta_info = [
            json.loads(line) for line in open(os.path.join(self.bench_path, 'metadata.jsonl'), 'r')
        ]
        items = {}
        for item in meta_info:
            items[item['key']] = {
                "instruction": item["instruction"],
                "input_image": os.path.join(self.bench_path, item["source_image"]),
                "edited_images": [
                    os.path.join(self.bench_path, img_path) for img_path in item['edited_images']
                ],
                "winner": item['winner'],
                "task_type": item['task'],
            }
        return items


def _load_model_candidates(model_lists: dict):
    candidates_dict = {}
    for model_name, model_results_jsonl_path in model_lists.items():
        candidates = {}
        resolved_path = str(resolve_project_path(model_results_jsonl_path))
        for _line in open(resolved_path, 'r'):
            line = json.loads(_line.strip())
            candidates[line['key']] = line['image_path']
        candidates_dict[model_name] = candidates
    return candidates_dict

class GEditBenchV2Eval(BaseDataset):
    def __init__(self, bench_path, meta_file, selected_compare_model_list):
        self.benchmark_name = DEFAULT_BENCHMARK_NAME
        self.bench_path = str(resolve_project_path(bench_path))
        self.meta_file = meta_file
        self.valid_compare_model_list = self._check_candidate_models(selected_compare_model_list)

        start_time = time.time()
        self.dataset = self.prepare_dataset()
        print(f"GEditBench v2 dataset loaded in {time.time() - start_time} seconds")

    def _check_candidate_models(self, selected_compare_model_list):
        print("| Model Name           | Number of Samples |")
        print("|----------------------|-------------------|")
        valid_models = []
        model_sample_counts = {}
        for model_name in selected_compare_model_list:
            if not os.path.exists(os.path.join(self.bench_path, 'images', 'edited', model_name)):
                raise ValueError(f"Model {model_name} does not exist in the benchmark edited images directory.")
            num_samples = len(os.listdir(os.path.join(self.bench_path, 'images', 'edited', model_name)))
            model_sample_counts[model_name] = num_samples
            print(f"{model_name:<20} | {num_samples}")
            if num_samples > 200:
                valid_models.append(model_name)
        for model_name in selected_compare_model_list:
            if model_name not in valid_models:
                print(
                    f"Warning: Model {model_name} has only {model_sample_counts[model_name]} samples, "
                    "which may lead to unreliable evaluation results. "
                    "Consider removing it from the selected_compare_model_list."
                )
        return valid_models        

    def prepare_dataset(self) -> Dict[str, Dict]:
        meta_info = [
            json.loads(line) for line in open(os.path.join(self.bench_path, self.meta_file), 'r')
        ]

        items = {}
        for item in meta_info:
            candidates_list = item['candidates']
            existing_image_paths = {
                candidate_model_info['model']: os.path.join(self.bench_path, candidate_model_info['image'])
                for candidate_model_info in candidates_list
                if candidate_model_info['model'] in self.valid_compare_model_list
            }
            pairs_dict = generate_canonical_pairs(existing_image_paths, seed=item['key'])
            for pair_key, pair_paths in pairs_dict.items():
                items[f"{item['key']}_pair_{pair_key}"] = {
                    "instruction": item["instruction"],
                    "input_image": os.path.join(self.bench_path, item["source_image"]),
                    "edited_images": pair_paths,
                }
        print(f"Loaded {len(items)} pairwise samples!")
        return items

class CandidatesDataset(BaseDataset):
    def __init__(self, data_config, pipeline_name):
        self.data_config = data_config
        self.pipelnie_name = pipeline_name

        start_time = time.time()
        self.dataset = self.prepare_dataset()
        print(f"Dataset loaded in {time.time() - start_time} seconds from {self.data_config['meta_info']}")

    def prepare_input_dicts_to_process(
        self,
        meta_info: Dict,
        candidates_dicts: Dict,
    ) -> Dict[str, Dict]:
        input_dicts = {}
        for item_key, item_meta_info in meta_info.items():
            existing_image_paths = {
                model_name: model_candidates_dict[item_key]
                for model_name, model_candidates_dict in candidates_dicts.items()
                if model_candidates_dict.get(item_key, None) is not None
            }
            if len(existing_image_paths) < 2:
                logger.info(f"Skipping item {item_key} due to insufficient image paths (< 2).")
                continue

            # prepare input dicts based on mode
            if self.pipelnie_name == 'vlm-as-a-judge':
                in_group_pairs = self.data_config.get("in_group_pairs", 6)
                pairs_dict = generate_canonical_pairs(existing_image_paths, n_pairs=in_group_pairs, seed=item_key)
                for pair_key, pair_paths in pairs_dict.items():
                    input_dicts[f"{item_key}_pair_{pair_key}"] = {
                        "instruction": item_meta_info['instruction'],
                        "input_image": item_meta_info['input_image'],
                        "edited_images": pair_paths,
                    }
            elif self.pipelnie_name in ['human-centric', 'object-centric']:
                for model_name, image_path in existing_image_paths.items():
                    input_dicts[f"{item_key}_model_{model_name}"] = {
                        "instruction": item_meta_info['instruction'],
                        "input_image": item_meta_info['input_image'],
                        "edited_images": [image_path],
                    }
            else:
                raise NotImplementedError(f"Mode {self.pipelnie_name} is not implemented in input dict preparation.")
        return input_dicts
    
    def prepare_dataset(self) -> Dict[str, Dict]:
        meta_info = {}
        meta_info_path = str(resolve_project_path(self.data_config['meta_info']))
        with open(meta_info_path, 'r') as f:
            for line in f:
                obj = json.loads(line.strip())
                meta_info[obj['key']] = {
                    'instruction': obj['instruction'],
                    'input_image': obj['image_path'],
                }
        candidates_dicts = _load_model_candidates(self.data_config['models'])
        input_dicts = self.prepare_input_dicts_to_process(meta_info, candidates_dicts)
        return input_dicts


def load_dataset(bmk_name: str, **kwargs):
    bmk_name = normalize_benchmark_name(bmk_name)
    if bmk_name in ["editscore_consistency", "editscore_prompt_following"]:
        assert kwargs['eval_dim'] in ['overall', 'consistency', 'prompt_following'], (
            "EditScore benchmark only supports 'overall', 'consistency', and 'prompt_following' evaluation dimensions."
        )
        print(f"Evaluating on EditScore benchmark for dimension: {kwargs['eval_dim']}...")
        return EditScoreRewardBenchmark(
            data_path=kwargs['data_path'],
            eval_dim=kwargs['eval_dim'],
            shuffled_res_path=kwargs['shuffled_res_path'],
            max_workers=kwargs.get('max_workers', None)
        )
    elif bmk_name == 'vc_reward':
        return VCRewardBenchmark(
            bench_path=kwargs['bench_path']
        )
    elif bmk_name == 'editreward_visual_quality':
        return EditRewardRewardBenchmark(
            image_data_path=kwargs['image_data_path'],
            json_file_path=kwargs['json_file_path']
        )
    elif bmk_name == 'candidates':
        return CandidatesDataset(
            data_config=kwargs['data_config'],
            pipeline_name=kwargs['pipeline_name']
        )
    elif bmk_name == DEFAULT_BENCHMARK_NAME:
        return GEditBenchV2Eval(
            bench_path=kwargs['bench_path'],
            meta_file=kwargs.get('meta_file', 'metadata.jsonl'),
            selected_compare_model_list=kwargs['selected_compare_model_list']
        )
    else:
        raise ValueError(f"Unsupported benchmark name: {bmk_name}")
