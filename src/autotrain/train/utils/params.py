import sys
import yaml
from pathlib import Path
from typing import Any, Optional, Union, Tuple, List, Literal
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class DataConfig:
    train_json_list: List[str] = field(
        default_factory=lambda: ["train_data.json"],
        metadata={"help": "Training data distribution json file."}
    )
    eval_json_list: List[str] = field(
        default_factory=lambda: ["eval_data.json"],
        metadata={"help": "List of evaluation data json files."}
    )
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    prompt_info: dict = field(default=None)


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class TrainingConfig(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    disable_dropout: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    head_lr: Optional[float] = None

    conduct_eval: Optional[bool] = True
    load_from_pretrained: str = None
    load_from_pretrained_step: int = None
    logging_epochs: Optional[float] = None
    eval_epochs: Optional[float] = None
    save_epochs: Optional[float] = None
    remove_unused_columns: Optional[bool] = False

    save_full_model: Optional[bool] = False
    
    # Visualization parameters
    # visualization_steps: Optional[int] = 100
    # max_viz_samples: Optional[int] = 4

    # PEFT LoRA Config
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target_modules: Optional[List[str]] = None
    lora_namespan_exclude: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    num_lora_modules: int = -1
    use_liger_kernel: bool = True




def parse_args_with_yaml(
    dataclass_types: Tuple[type, ...], 
    config_path: str = None,
    allow_extra_keys: bool = True,
    is_train: bool = True,
) -> Tuple[Any, ...]:
    """
    Parse arguments using HfArgumentParser with OmegaConf for YAML support.
    
    Args:
        dataclass_types: Tuple of dataclass types for HfArgumentParser
        args: Optional arguments (if None, will read from sys.argv)
        allow_extra_keys: Whether to allow extra keys in config
    
    Returns:
        Tuple of parsed dataclass instances
    """
    # Read arguments from command line or provided args
    # Load YAML config and merge with command line overrides
    args = OmegaConf.to_container(OmegaConf.load(config_path))
    if not is_train:
        args.pop('deepspeed', None)

    # Parse with HfArgumentParser
    parser = HfArgumentParser(dataclass_types)
    return parser.parse_dict(args, allow_extra_keys=allow_extra_keys), config_path