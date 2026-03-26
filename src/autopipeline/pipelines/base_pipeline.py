from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from ..components import PIPE_REGISTRY, EXPERT_REGISTRY
from src.prompts.prompt_manager import PromptAssetStore
from ..components.modules.judge import ClientPipe
from common_utils.image_util import open_image, compress_convert_image2any
from common_utils.logging_util import get_logger
logger = get_logger()
SCOPE_TO_MASK_MODE = {
    'unedit_area': 'outer',
    'edit_area': 'inner',
}
REGIONS_TO_EXPERTS_MAP = {
    'body': ["human-segmenter", "hair-segmenter"],
    'hair': ["hair-segmenter"],
    'face': ["face-detector"],
}
DOUBLE_AREA_SCALE_TASK = [
    'PS_HUMAN', 'MOTION_CHANGE', 'SUBJECT_REPLACE', 'MOTION_CHANGE',
    'OBJECT_EXTRACTION', 'OREF', 'SIZE_ADJUSTMENT'
]

@dataclass(frozen=True)
class PipeKey:
    pipe_name: str
    init_config_key: Tuple

def freeze_config(cfg):
    if not cfg:
        return ()
    frozen_items = []
    for k, v in sorted(cfg.items()):
        if isinstance(v, dict):
            v = freeze_config(v)
        elif isinstance(v, list):
            v = tuple(v)
        frozen_items.append((k, v))
    return tuple(frozen_items)

class PipeRegistry:
    def __init__(self):
        self._pipes = {}

    def get(self, pipe_key: PipeKey, prompt_manager: PromptAssetStore = None):
        if pipe_key not in self._pipes:
            pipe_cls = PIPE_REGISTRY.get(pipe_key.pipe_name)
            if not pipe_cls:
                raise ValueError(f"Pipe '{pipe_key.pipe_name}' is not registered.")
            init_config = dict(pipe_key.init_config_key)
            
            if issubclass(pipe_cls, ClientPipe):
                try:
                    prompt_template = prompt_manager.get_prompt(**dict(init_config['prompt_info']))
                    self._pipes[pipe_key] = pipe_cls(prompt_template=prompt_template, client_cfg=init_config)
                    return self._pipes[pipe_key]
                except Exception as e:
                    raise ValueError(f"ClientPipe '{pipe_key.pipe_name}' requires a PromptAssetStore instance: {e}")
            
            self._pipes[pipe_key] = pipe_cls(**init_config)
        return self._pipes[pipe_key]


class BasePipeline(ABC):
    required_configs: List[str] = []

    def __init__(self, pipeline_config):
        self.logger = logger
        self.pipe_registry = PipeRegistry()
        self.prompt_manager = PromptAssetStore()
        self.pipeline_config = pipeline_config
        self._validate_config()

    def _validate_config(self):
        missing = [k for k in self.required_configs if k not in self.pipeline_config]
        if missing:
            raise ValueError(
                f"Missing required configs {missing} for `{self.__class__.__name__}`."
            )

    def _load_parser_grounder_module(self, parser_grounder_config):
        parser_grounder_cls = PIPE_REGISTRY.get("parser-grounder")
        parser_grounder = parser_grounder_cls(
            parser_grounder_config,
            {
                "instruction_parser": self.prompt_manager.get_prompt(
                    **dict(dict(parser_grounder_config['instruction_parser']['init_config'])['prompt_info'])
                ),
                "general_grounder": self.prompt_manager.get_prompt(
                    **dict(dict(parser_grounder_config['general_grounder']['init_config'])['prompt_info'])
                ),
            }
        )

        return parser_grounder
    
    def _load_expert_module(self, expert_configs, metric_configs):
        # check required segmenters
        self.required_experts = set()
        for metric in metric_configs.keys():
            required_expert = REGIONS_TO_EXPERTS_MAP.get(metric.split('_')[0], [])
            if required_expert:
                self.required_experts.update(required_expert)
        # load expert modules
        experts = {}
        for expert_name in self.required_experts:
            if expert_name in expert_configs.keys():
                expert_cls = EXPERT_REGISTRY.get(expert_name)
                experts[expert_name] = expert_cls(**dict(expert_configs[expert_name]))
            else:
                raise ValueError(f"Metric configs require expert '{expert_name}' but it's not found in expert_configs.")
        return experts

    def parse_image_info(self, image_path, max_side=None, target_type=None):
        return compress_convert_image2any(
            open_image(image_path),
            max_side=max_side,
            target_type=target_type
        )

    def parse_metric_configs(self, metric_configs):
        metric_to_pipekey = {}
        pipekey_to_metrics = {}
        for metric_name, cfg in metric_configs.items():
            pipe_key = PipeKey(
                pipe_name=cfg['pipe_name'],
                init_config_key=freeze_config(cfg.get('init_config', {}))
            )
            metric_to_pipekey[metric_name] = {
                "pipe_key": pipe_key,
                "scope": cfg.get('scope', None),
                "runtime_params": cfg.get('runtime_params', {}),
            }
            pipekey_to_metrics.setdefault(pipe_key, []).append(metric_name)
        # self.logger.info(f"📋 Parsed {len(metric_to_pipekey)} metrics and {len(pipekey_to_metrics)} unique pipes from config.")
        return metric_to_pipekey, pipekey_to_metrics
    
    def smart_load_pipes(self, metric_to_pipekey):
        metric_pipe_dict = {}
        for metric_name, info in metric_to_pipekey.items():
            pipe = self.pipe_registry.get(info['pipe_key'], self.prompt_manager)
            metric_pipe_dict[metric_name] = {
                "pipe": pipe,
                "scope": info['scope'],
                "runtime_params": info['runtime_params'],
                "mask_mode": SCOPE_TO_MASK_MODE.get(info['scope'], None),
            }
        return metric_pipe_dict

    def compute_edited_area_ratio(self, image_size, coords, edit_task_type):
        scale_factor = 2 if edit_task_type in DOUBLE_AREA_SCALE_TASK else 1
        image_area = image_size[0] * image_size[1] * scale_factor
        edited_area = 0
        for box in coords:
            box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            edited_area += box_area
        edited_area_ratio = edited_area / image_area
        return edited_area_ratio

    @abstractmethod
    def __call__(self, input_dict, **kwargs):
        pass