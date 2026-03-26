import os
import json
import torch
import pandas as pd
from peft import LoraConfig, get_peft_model
import ast
import fire
from dataclasses import asdict
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)
from autotrain.train.qwen_vl_trainer import QwenSFTTrainer
from autotrain.train.dataset import make_pairwise_data_module
from autotrain.train.utils.params import DataConfig, ModelConfig, TrainingConfig
from autotrain.train.utils.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
# from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from autotrain.train.utils.monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward, 
    replace_qwen_2_with_mixed_modality_forward
)
from autotrain.train.utils.monkey_patch_vision import replace_qwen2_5_vision
from autotrain.train.utils.params import parse_args_with_yaml
from autotrain.constants import JSON_CONFIG_SAVE_PATH

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(model.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = model.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True

def save_configs_to_json(data_config, training_args, model_config):
    """
    Save all configurations to a JSON file.
    """
    config_dict = {
        "data_config": asdict(data_config),
        "training_args": asdict(training_args),
        "model_config": asdict(model_config),
    }
    # del information about local device
    del config_dict["training_args"]["local_rank"]
    del config_dict["training_args"]["_n_gpu"]

    save_path = os.path.join(
        JSON_CONFIG_SAVE_PATH,
        pd.Timestamp.now().strftime("%m%d"),
    )
    save_file = f"sft_config_{pd.Timestamp.now().strftime('%H%M%S')}.json"

    os.makedirs(save_path, exist_ok=True)
    print(f"⭐ Config file will be stored to: {save_path} as {save_file} for checking.")

    with open(f"{save_path}/{save_file}", "w") as f:
        json.dump(config_dict, f, indent=4)


def train(config, output_dir=None):
    global local_rank

    (data_config, training_args, model_config), config_path = (
        parse_args_with_yaml(
            (DataConfig, TrainingConfig, ModelConfig), config, is_train=True
        )
    )
    # training_args.output_dir = os.path.join(
    #     training_args.output_dir,
    #     f"{config.split("/")[-1].split(".")[0]}"
    # )
    if output_dir is not None:
        training_args.output_dir = output_dir
    training_args.logging_dir = os.path.join(
        "logs/training_loss_tb_logs",
        training_args.output_dir.split('/')[-2],
        training_args.output_dir.split('/')[-1]
    )

    if "Qwen2.5" in model_config.model_name_or_path:
        # monkey patch the vision model
        replace_qwen2_5_vision()
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward()
    elif "Qwen3" in model_config.model_name_or_path:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen3_with_mixed_modality_forward()
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward()

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")
    
    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
    
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")
    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    
    if "Qwen2.5" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    elif "Qwen3" in model_config.model_name_or_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )
        
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    data_module = make_pairwise_data_module(
        model_name_or_path=model_config.model_name_or_path,
        processor=processor,
        data_config=data_config
    )

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        save_configs_to_json(data_config, training_args, model_config)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    fire.Fire(train)
