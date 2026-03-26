import torch
from PIL import Image
from typing import List, Tuple
from ..constant import EMBED_MODEL_RESOLUTION
from common_utils.logging_util import get_logger
logger = get_logger()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class CLIPMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoProcessor, CLIPVisionModel
        model_path = kwargs.get('model_path', 'openai/clip-vit-base-patch32')
        self.img_input_size, self.patch_size = EMBED_MODEL_RESOLUTION.get(model_path.split('/')[-1], (224, 32))
        self.device = kwargs.get('device', DEVICE)
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = CLIPVisionModel.from_pretrained(model_path).to(self.device)

        logger.info(f"CLIP model loaded: {model_path}, img_input_size: {self.img_input_size}, patch_size: {self.patch_size}")

    def get_features(self, image: Image.Image, mask=None) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 1:, :].detach().squeeze(0) # exclude CLS token
        return embeddings[mask, :] if mask is not None else embeddings

    def get_cls_feature(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().squeeze(0) # CLS token
        return cls_embedding


class DINOv3Mixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoImageProcessor, AutoModel
        model_path = kwargs.get('model_path', None)
        self.input_image_size, self.patch_size = EMBED_MODEL_RESOLUTION.get(model_path.split('/')[-1], (224, 16))
        self.device = kwargs.get('device', DEVICE)

        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
    
    def get_features(self, image: Image.Image, mask=None) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        special_token_num = 1 + self.model.config.num_register_tokens  # +1 for CLS token
        embeddings = outputs.last_hidden_state[:, special_token_num:, :].detach().squeeze(0)  # (num_patches, feature_dim)
        return embeddings[mask, :] if mask is not None else embeddings
    
    def get_cls_feature(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().squeeze(0)  # CLS token
        return cls_embedding