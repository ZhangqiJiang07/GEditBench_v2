import os
import sys
sys.path.append('/data/open_edit/src')
print(sys.path)
import json
import random
import torch
from torch.utils.data import Dataset
import transformers
from typing import Dict, List, Any
from autotrain.train.utils.params import DataConfig

from qwen_vl_utils import process_vision_info
from common_utils.image_util import check_image_exists, open_image
from prompts.prompt_manager import PromptAssetStore
from core.wrapper import ImageWrapper

from autotrain.train.utils.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_END_TOKEN,
    SEPARATOR_RULES,
)

# from autotrain.train.utils.prompt_base import (
#     ASSESSMENT_TYPE_TO_PROMPT_DICT,
#     # ASSESSMENT_TYPE_TO_PROMPT_DICT_WITH_REASON,
# )

def get_separator(prev_type, current_type):
    return SEPARATOR_RULES.get((prev_type, current_type), "\n")

class PairWiseDataset(Dataset):
    """ Dataset for pairwise comparison format."""
    def __init__(self,
        dataset_list: List[str],
        model_name_or_path: str,
        processor: transformers.ProcessorMixin,
        data_config: DataConfig,
        padding=False,
    ):
        super(PairWiseDataset, self).__init__()
        self.samples = []
        for data in dataset_list:
            self.samples.extend(
                json.load(open(data, 'r'))
            )

        self.processor = processor
        self.data_config = data_config
        self.model_name_or_path = model_name_or_path
        self.image_min_pixels = data_config.image_min_pixels
        self.image_max_pixels = data_config.image_max_pixels
        self.image_resized_width = data_config.image_resized_width
        self.image_resized_height = data_config.image_resized_height
        self.padding = padding

        # Initialize prompt templates
        self.prompt_info = data_config.prompt_info
        prompt_manager = PromptAssetStore(self.prompt_info.get("assets_dir"))
        self.prompt_template = prompt_manager.get_prompt(
            prompt_id=self.prompt_info.get("prompt_id"),
            version=self.prompt_info.get("version")
        )

        if "Qwen3" in model_name_or_path:
            self.image_patch_size = 16
        else:
            self.image_patch_size = 14
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        while True:
            index = idx
            try:
                sample_info = self._get_item_info(index)
                if sample_info is None:
                    return self.__getitem__((idx + 1) % len(self.samples))
                return self._parse_item_info(sample_info)

            except Exception as e:
                print(f"Failed to fetch sample {idx}. Exception:", e)
                import traceback
                traceback.print_exc()
                index = random.randint(0, len(self.samples) - 1)
                if index == idx:
                    continue
                idx = index

    def _get_item_info(self, idx) -> Dict[str, Any]:
        try:
            sample = self.samples[idx]

            source_image_path = sample.get("source_image_path")
            edited_image_paths = sample.get("edited_image_paths")
            gpt_response = sample.get("gpt_response") # first second tie

            if not (source_image_path and edited_image_paths and gpt_response):
                return None
            if not (check_image_exists(source_image_path) and
                    all(check_image_exists(p) for p in edited_image_paths)):
                return None

            instruction = sample.get("instruction", "")

            return {
                "input_image": open_image(source_image_path),
                "edited_images": [open_image(p) for p in edited_image_paths],
                "gpt_response": gpt_response,
                "instruction": instruction
            }

        except Exception as e:
            print(f"[WARN] Skipping idx={idx}, reason: {e}")
            return None
        
    def _prepare_message(self, **kwargs):
        system_prompt = self.prompt_template.system_prompt
        user_blocks = self.prompt_template.render_blocks(**kwargs)
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
                    "min_pixels": self.image_min_pixels,
                    "max_pixels": self.image_max_pixels,
                })
            # [block separator]
            if i < len(user_blocks) - 1:
                next_block = user_blocks[i + 1]
                separator = get_separator(block["type"], next_block["type"])
                user_content.append({"type": "text", "text": separator})

        if self.image_resized_width is not None and self.image_resized_height is not None:
            for content in user_content:
                if content.get("type", None) == "image":
                    content["resized_width"] = self.image_resized_width
                    content["resized_height"] = self.image_resized_height

        messages.append({"role": "user", "content": user_content})
        return messages
        
    def _parse_item_info(self, item_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Parse item info to input_ids.
        """
        all_pixel_values = []
        all_image_grid_thw = []
        grid_key = "image_grid_thw"
        pixel_key = "pixel_values"

        # messages = self._prepare_message(
        #     item_info["instruction"],
        #     item_info["source_image_path"],
        #     item_info["image1_path"],
        #     item_info["image2_path"]
        # )
        messages = self._prepare_message(**item_info)
        # print(f"[DEBUG] messages: {messages}")
        gpt_response = f"{item_info['gpt_response']}{DEFAULT_IM_END_TOKEN}\n"

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            add_vision_id=False
        )
        print(text + "abc")
        images, _ = process_vision_info(messages, image_patch_size=self.image_patch_size)
        # image_inputs = [np.array(x) / 255.0 for x in image_inputs]

        inputs = self.processor(
            text=[text],
            images=images,
            padding=False,
            return_tensors="pt",
            videos=None,
            do_resize=False
        )

        prompt_input_ids = inputs['input_ids']
        all_pixel_values.append(inputs[pixel_key])
        all_image_grid_thw.append(inputs[grid_key])

        # print("abc" + gpt_response)
        response_input_ids = self.processor.tokenizer(
            gpt_response,
            add_special_tokens=False,
            padding=False,
            return_tensors='pt'
        )['input_ids']
        
        input_ids = torch.cat(
            [
                prompt_input_ids,
                response_input_ids
            ],
            dim=1
        ).squeeze(0).to(torch.long)
        labels = torch.cat(
            [
                torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                response_input_ids.squeeze(0)
            ],
            dim=0
        ).to(torch.long)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        return data_dict


class DataCollatorForPairWiseDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def _pad_sequence(self, sequences, padding_side='right', padding_value=0):
        """
        Pad a list of sequences to the same length.
        sequences: list of tensors in [seq_len, *] shape
        """
        assert padding_side in ['right', 'left']
        max_size = sequences[0].size()
        trailing_dims = max_size[1:]
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)
        output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
        for i, seq in enumerate(sequences):
            length = seq.size(0)
            if padding_side == 'right':
                output.data[i, :length] = seq
            else:
                output.data[i, -length:] = seq

        return output

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_thw = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

        input_ids = self._pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = self._pad_sequence(
            batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX
        )

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        return data_dict
    

def make_pairwise_data_module(
    model_name_or_path,
    processor,
    data_config
):
    """Make dataset and collator for supervised fine-tuning."""
    pairwise_dataset = PairWiseDataset(
        dataset_list=data_config.train_json_list,
        model_name_or_path=model_name_or_path,
        processor=processor,
        data_config=data_config,
    )
    eval_dataset = None

    if data_config.eval_json_list is not None:
        eval_dataset = PairWiseDataset(
            dataset_list=data_config.eval_json_list,
            model_name_or_path=model_name_or_path,
            processor=processor,
            data_config=data_config,
        )

    data_collator = DataCollatorForPairWiseDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(train_dataset=pairwise_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


if __name__ == "__main__":
    from autotrain.train.utils.params import DataConfig
    from transformers import AutoProcessor

    data_config = DataConfig()
    data_config.train_json_list = [
        "/data/assess_model/data/consistency/EditReward_Data/overall_pair_data_00_of_08.json"
    ]
    data_config.image_min_pixels = 256 * 28 * 28
    data_config.image_max_pixels = 256 * 28 * 28
    data_config.image_resized_width = 512
    data_config.image_resized_height = 512
    data_config.prompt_info = {
        "assets_dir": "assets",
        "prompt_id": "vlm/assessment/visual_consistency/pairwise",
        "version": "v1"
    }
    # data_config.with_reason = False

    model_name_or_path = '/mnt/jfs/model-zoo/Qwen/Qwen2.5-VL-7B-Instruct'
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    dataset = PairWiseDataset(
        dataset_list=data_config.train_json_list,
        model_name_or_path=model_name_or_path,
        processor=processor,
        data_config=data_config,
    )

    for i in range(len(dataset)):
        item = dataset[i]
        # print(item)
        break