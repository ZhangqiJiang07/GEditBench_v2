import torch
from peft import PeftModel
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor
)

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def merge_lora(base_model_path, lora_model_path, save_dir):
    model_name = get_model_name_from_path(base_model_path)
    if "Qwen2.5" in model_name:
        print('Loading Qwen2.5-VL from base model...')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path, dtype=torch.bfloat16, device_map="cpu"
        )
    elif "Qwen3" in model_name:
        print('Loading Qwen3-VL from base model...')
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path, dtype=torch.bfloat16, device_map="cpu"
        )
    else:
        print('Loading Qwen2-VL from base model...')
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path, dtype=torch.bfloat16, device_map="cpu"
        )
    model = PeftModel.from_pretrained(model, lora_model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    model.save_pretrained(save_dir)
    
    processor = AutoProcessor.from_pretrained(base_model_path, fix_mistral_regex=True)
    processor.save_pretrained(save_dir)    
    print(f'Model and processor successfully saved to {save_dir}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--lora-weights-path", type=str, required=True)
    parser.add_argument("--model-save-dir", type=str, default="./merged_model")
    args = parser.parse_args()
    
    merge_lora(args.base_model_path, args.lora_weights_path, args.model_save_dir)