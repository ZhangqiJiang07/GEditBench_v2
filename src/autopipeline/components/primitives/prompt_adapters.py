from . import PROMPT_ADAPTER_REGISTRY
from google.genai import types
from abc import ABC, abstractmethod
from core.wrapper import ImageWrapper

SEPARATOR_RULES = {
    ("text", "image"): "", # \n
    ("image", "text"): "\n", # \n\n
    ("image", "image"): "\n\n",
}

def get_separator(prev_type, current_type):
    return SEPARATOR_RULES.get((prev_type, current_type), "\n")

class BasePromptAdapter(ABC):
    @abstractmethod
    def build_payload(self, template, **kwargs) -> list:
        pass

@PROMPT_ADAPTER_REGISTRY.register("openai_style")
class OpenAIStylePromptAdapter(BasePromptAdapter):
    def build_payload(self, template, **kwargs):
        system_prompt = template.system_prompt
        user_blocks = template.render_blocks(**kwargs)
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
                    "image_url": {"url": img_wrapper.as_data_url()}
                })
            
            # [block separator]
            if i < len(user_blocks) - 1:
                next_block = user_blocks[i + 1]
                separator = get_separator(block["type"], next_block["type"])
                user_content.append({"type": "text", "text": separator})

        messages.append({"role": "user", "content": user_content})
        # print(f"[DEBUG] OpenAIStylePromptAdapter messages: {messages}")
        return messages


@PROMPT_ADAPTER_REGISTRY.register("google_style")
class GoogleGenAIStylePromptAdapter(BasePromptAdapter):
    def build_payload(self, template, **kwargs):
        system_prompt = template.system_prompt
        user_blocks = template.render_blocks(**kwargs)
        parts = []

        first_text_block_flag = 1
        for i, block in enumerate(user_blocks):
            if block["type"] == "text":
                if system_prompt and first_text_block_flag:
                    parts.append(f"{system_prompt}\n\n{block['content']}")
                    first_text_block_flag = 0
                else:
                    parts.append(block['content'])
            elif block["type"] == "image":
                img_wrapper: ImageWrapper = block["content"]
                parts.append(types.Part.from_bytes(
                    data=img_wrapper.as_bytes(),
                    mime_type='image/png'
                ))

            # [block separator]
            if i < len(user_blocks) - 1:
                next_block = user_blocks[i + 1]
                separator = get_separator(block["type"], next_block["type"])
                parts.append(separator)
        return parts
                

