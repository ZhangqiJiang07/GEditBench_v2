import re
import jinja2
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from core.wrapper import ImageWrapper

class PromptTemplate(BaseModel):
    prompt_id: str = Field(..., description="Unique identifier for the prompt template, e.g., 'assessment/camera_motion_visual_consistency'")
    version: str = Field("v1", description="Version of the prompt template, e.g., 'v1'")
    system_prompt: str = Field(..., description="The system prompt content")
    user_prompt: List[Dict[str, Any]] = Field(..., description="Interleaved user prompt content blocks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the prompt template")

    def render_blocks(self, **kwargs) -> str:
        """
        将 YAML 中的结构化定义，结合传入的kwargs，渲染成标准的底层block
        """
        blocks = []
        for item in self.user_prompt:
            if "text" in item:
                jinja_template = jinja2.Template(item["text"])
                rendered_text = jinja_template.render(**kwargs)
                blocks.append({"type": "text", "content": rendered_text})

            elif "image" in item:
                img_info = item["image"]
                source_key = img_info.get("source")

                if source_key not in kwargs:
                    raise ValueError(f"Missing required image source for key '{source_key}' in kwargs")
                img_data = kwargs[source_key]

                if "index" in img_info:
                    idx = img_info["index"]
                    if not isinstance(img_data, list) or idx >= len(img_data):
                        raise ValueError(f"Invalid index '{idx}' for image source '{source_key}'")
                    final_image = img_data[idx]
                else:
                    final_image = img_data
                blocks.append({"type": "image", "content": ImageWrapper(final_image)})

        return blocks