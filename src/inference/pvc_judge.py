import os
import torch
import time
from peft import PeftModel
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info
from inference.utils.lora_merge import get_model_name_from_path
from common_utils.json_util import extract_winner_from_text
from common_utils.image_util import open_image
from schemas.prompt_template import PromptTemplate
from core.wrapper import ImageWrapper
from autotrain.train.utils.constants import SEPARATOR_RULES
from common_utils.logging_util import get_logger
logger = get_logger()

def get_separator(prev_type, current_type):
    return SEPARATOR_RULES.get((prev_type, current_type), "\n")


class BasicPairWiseJudge:
    def __init__(
        self,
        lora_model_path: str,
        base_model_path: str,
        prompt_template: PromptTemplate,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_root = kwargs.get("cache_root", None)
        cache_dir = self.make_cache_dir(lora_model_path, cache_root=cache_root)
        if not os.path.exists(cache_dir):
            self._merge_lora_and_save_model(
                lora_model_path=lora_model_path,
                base_model_path=base_model_path,
                use_flash_attn=use_flash_attn,
                cache_dir=cache_dir,
                **kwargs
            )
        self._load_merged_model(cache_dir)
        self.prompt_template = prompt_template

    def make_cache_dir(self, lora_model_path: str, cache_root=None) -> str:
        if cache_root is None:
            cache_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_dir = os.path.join(
            cache_root,
            f"_".join([
                f"{lora_model_path.split('/')[-3]}",
                f"{lora_model_path.split('/')[-2]}",
                f"{lora_model_path.split('/')[-1]}",
            ]).replace("-", "_")
        )
        return cache_dir

    def _merge_lora_and_save_model(
            self,
            lora_model_path: str,
            base_model_path: str,
            use_flash_attn: bool = False,
            cache_dir: str = "",
            **kwargs
        ):
        start_time = time.time()
        model_name = get_model_name_from_path(base_model_path)
        logger.info(f"Merging LoRA weights into base model for {model_name}...")
        if "Qwen2.5" in model_name:
            logger.info('Loading Qwen2.5-VL from base model...')
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path, dtype=torch.bfloat16, device_map="cpu"
            )
        elif "Qwen3" in model_name:
            logger.info('Loading Qwen3-VL from base model...')
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path, dtype=torch.bfloat16, device_map="cpu"
            )
        else:
            logger.info('Loading Qwen2-VL from base model...')
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_path, dtype=torch.bfloat16, device_map="cpu"
            )
        model = PeftModel.from_pretrained(model, lora_model_path)
        logger.info('Merging LoRA weights...')
        model = model.merge_and_unload()
        model.save_pretrained(cache_dir)

        processor = AutoProcessor.from_pretrained(base_model_path, fix_mistral_regex=True) # fix_mistral_regex=True 加不加都不影响最后的结果
        processor.save_pretrained(cache_dir)                                               # 主要是处理issue: The tokenizer you are loading from 'xxxx' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
        
        logger.info(f"Merged LoRA model saved to {cache_dir} took {time.time() - start_time} seconds!!!")

    def _load_merged_model(self, model_path: str):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _parse_input_dict(self, input_dict):
        return {
            "input_image": open_image(input_dict["input_image"]),
            "edited_images": [open_image(p) for p in input_dict["edited_images"]],
            "instruction": input_dict["instruction"]
        }

    def prepare_messages(self, input_dict, generation_config):
        image_min_pixels = generation_config.get("image_min_pixels", 256 * 32 * 32)
        image_max_pixels = generation_config.get("image_max_pixels", 1280 * 32 * 32)
        parsed_input = self._parse_input_dict(input_dict)

        system_prompt = self.prompt_template.system_prompt
        user_blocks = self.prompt_template.render_blocks(**parsed_input)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        for i, block in enumerate(user_blocks):
            if block["type"] == "text":
                user_content.append({"type": "text", "text": block["content"]})
            elif block["type"] == "image":
                img_wrapper: ImageWrapper = block["content"]
                user_content.append({
                    "type": "image_url",
                    "image_url": img_wrapper.as_data_url(),
                    "min_pixels": image_min_pixels,
                    "max_pixels": image_max_pixels,
                })
            if i < len(user_blocks) - 1:
                next_block = user_blocks[i + 1]
                separator = get_separator(block["type"], next_block["type"])
                user_content.append({"type": "text", "text": separator})

        messages.append({"role": "user", "content": user_content})
        return messages

    def __call__(self, input_dict, generation_config: dict, **kwargs):
        messages = self.prepare_messages(input_dict, generation_config)
        images, _ = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            add_vision_id=False
        )

        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        max_retries = generation_config.get('max_retries', 10)

        winner = None
        for i in range(max_retries):
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config.get("max_new_tokens", 512),
                    num_beams=generation_config.get("num_beams", 1),
                    do_sample=generation_config.get("do_sample", False),
                    temperature=generation_config.get("temperature", 1.0)
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # print("Generated Text:", output_text[0])
            winner = extract_winner_from_text(output_text[0])
            if winner is not None:
                break
            else:
                logger.debug(f"➰ Retry {i+1}/{max_retries} to get valid winner...")

        return winner, {"winner": winner, "generated_text": output_text[0]}
    

class vLLMPairWiseJudge(BasicPairWiseJudge):
    def __init__(
        self,
        lora_model_path,
        base_model_path,
        prompt_template: PromptTemplate,
        use_flash_attn=False,
        **kwargs
    ):
        logger.info("Initializing vLLM-based PairWise Judge...")
        self.max_model_len = kwargs.get("max_model_len", 8192)
        self.max_num_seqs = kwargs.get("max_num_seqs", 32)
        self.max_num_batched_tokens = kwargs.get("max_num_batched_tokens", 15360)
        self.prompt_template = prompt_template
        cache_root = kwargs.get("cache_root", None)
        cache_dir = self.make_cache_dir(lora_model_path, cache_root=cache_root)
        if not os.path.exists(cache_dir):
            self._merge_lora_and_save_model(
                lora_model_path=lora_model_path,
                base_model_path=base_model_path,
                use_flash_attn=use_flash_attn,
                cache_dir=cache_dir,
                **kwargs
            )
        logger.info("Cached model found, loading...")
        self._load_merged_model(cache_dir)
        logger.info("vLLM PairWise Judge Loaded!")

    # def _merge_lora_and_save_model(
    #         self,
    #         lora_model_path: str,
    #         base_model_path: str,
    #         use_flash_attn: bool = False,
    #         cache_dir: str = "",
    #         **kwargs
    #     ):
    #     start_time = time.time()
    #     model_name = get_model_name_from_path(base_model_path)
    #     logger.info(f"Merging LoRA weights into base model for {model_name}...")
    #     if "Qwen2.5" in model_name:
    #         logger.info('Loading Qwen2.5-VL from base model...')
    #         model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #             base_model_path, dtype=torch.bfloat16, device_map="cpu"
    #         )
    #     elif "Qwen3" in model_name:
    #         logger.info('Loading Qwen3-VL from base model...')
    #         model = Qwen3VLForConditionalGeneration.from_pretrained(
    #             base_model_path, dtype=torch.bfloat16, device_map="cpu"
    #         )
    #     else:
    #         logger.info('Loading Qwen2-VL from base model...')
    #         model = Qwen2VLForConditionalGeneration.from_pretrained(
    #             base_model_path, dtype=torch.bfloat16, device_map="cpu"
    #         )
    #     model = PeftModel.from_pretrained(model, lora_model_path)
    #     logger.info('Merging LoRA weights...')
    #     model = model.merge_and_unload()
    #     model.save_pretrained(cache_dir)

    #     processor = AutoProcessor.from_pretrained(base_model_path, fix_mistral_regex=True) # fix_mistral_regex=True 加不加都不影响最后的结果
    #     processor.save_pretrained(cache_dir)                                               # 主要是处理issue: The tokenizer you are loading from 'xxxx' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
        
    #     logger.info(f"Merged LoRA model saved to {cache_dir} took {time.time() - start_time} seconds!!!")

    def _load_merged_model(self, model_path: str):
        logger.info(f"Loading VLLM model from {model_path}...")
        from vllm import LLM
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 8:
            tp_size = 8
        elif gpu_count >= 4:
            tp_size = 4
        elif gpu_count >= 2:
            tp_size = 2
        else:
            tp_size = 1
        logger.debug(f"Detected {gpu_count} GPUs, setting tensor parallel size to {tp_size}.")

        self.model = LLM(
            model=model_path,
            max_model_len=self.max_model_len,
            tensor_parallel_size=tp_size,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            limit_mm_per_prompt={"image": 10},
            enable_prefix_caching=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, input_dict, generation_config: dict, **kwargs):
        from vllm.sampling_params import SamplingParams
        max_new_tokens = generation_config.get("max_new_tokens", 512)
        top_p = generation_config.get("top_p", 0.9)
        top_k = generation_config.get("top_k", 20)
        temperature = generation_config.get("temperature", 1.0)
        # repetition_penalty = generation_config.get("repetition_penalty", 1.0)
        seed = generation_config.get("seed", 42)
        
        messages = self.prepare_messages(input_dict, generation_config)
        image_inputs, _ = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        llm_inputs = {
            "prompt": self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                add_vision_id=False
            ),
            "multi_modal_data": mm_data,
        }

        winner_candidates = []
        generated_text_record = []
        max_retries = generation_config.get('max_retries', 10)
        num_pass = generation_config.get("num_pass", 1)
        for pass_count in range(1, num_pass + 1):
            logger.debug(f"--- Verification Pass {pass_count} ---")
            pass_winner = None
            for i in range(max_retries):
                sampling_params = SamplingParams(
                    max_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    # repetition_penalty=repetition_penalty,
                    seed=(seed + i) * pass_count,
                )
                outputs = self.model.generate(
                    [llm_inputs],
                    sampling_params,
                    use_tqdm=False,
                )
                generated_text = outputs[0].outputs[0].text.strip()
                logger.debug("Generated Text: " + generated_text)
                pass_winner = extract_winner_from_text(generated_text)
                if pass_winner is not None:
                    break
                else:
                    logger.debug(f"➰ Retry {i+1}/{max_retries} to get valid winner...")
            if pass_winner is None:
                logger.warning(f"Failed to get valid winner after {max_retries} retries in pass {num_pass}.")
            else:
                winner_candidates.append(pass_winner)
                generated_text_record.append(generated_text)
                logger.debug(f"Pass {pass_count} winner: {pass_winner}")
        # Majority voting
        if len(winner_candidates) == 0:
            winner = None
        else:
            winner = max(set(winner_candidates), key=winner_candidates.count)
            logger.debug(f"Final winner after {generation_config.get('num_pass', 1)} passes: {winner} with votes: {winner_candidates.count(winner)}/{len(winner_candidates)}")

        return winner, {"winner": winner, "generated_text": generated_text}
    

class PairWiseJudge:
    def __init__(self, use_vllm: bool, **kwargs):
        if use_vllm:
            self.pvc_judge = vLLMPairWiseJudge(**kwargs)
        else:
            self.pvc_judge = BasicPairWiseJudge(**kwargs)

    def __call__(self, input_dict, generation_config: dict, **kwargs):
        return self.pvc_judge(input_dict, generation_config, **kwargs)