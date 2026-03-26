import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import EXPERT_REGISTRY


@EXPERT_REGISTRY.register("human-segmenter")
class HumanSegmentationMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=kwargs.get('model_selection', 1)
        )
        self.seg_threshold = kwargs.get('threshold', 0.5)

    def get_mask(self, image: Image.Image) -> np.ndarray:
        image_rgb = np.array(image.convert("RGB"))
        h_img, w_img = image_rgb.shape[:2]

        results = self.segmenter.process(image_rgb)
        if not results.segmentation_mask.any():
            return None

        seg_mask = results.segmentation_mask
        bool_mask = seg_mask > self.seg_threshold
        human_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        human_mask[bool_mask] = 1

        return human_mask


@EXPERT_REGISTRY.register("hair-segmenter")
class HairSegmentationMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = kwargs.get('model_path', None)
        if self.model_path is None or not os.path.exists(self.model_path):
            raise ValueError("Valid model_path must be provided for HairSegmentationMixin. Download from [https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter?hl=zh-cn#hair-model]")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )

        self.hair_segmenter = vision.ImageSegmenter.create_from_options(options)
    
    def _convert_to_mp_image(self, image_input):
        """辅助函数：转为 mp.Image"""
        if isinstance(image_input, Image.Image):
            # PIL (RGB) -> Numpy (RGB)
            img_np = np.array(image_input.convert('RGB'))
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        
        elif isinstance(image_input, np.ndarray):
            # Numpy (BGR assumed from cv2) -> RGB
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_input)
        else:
            raise TypeError("Unsupported image type")

    def get_mask(self, image: Image.Image) -> np.ndarray:
        image_rgb = np.array(image.convert("RGB"))
        h_img, w_img = image_rgb.shape[:2]

        mp_image = self._convert_to_mp_image(image)
        result = self.hair_segmenter.segment(mp_image)

        if result.category_mask is None:
            return None

        category_mask = result.category_mask
        mask_np = category_mask.numpy_view()
        bool_mask = mask_np > 0
        # print('🚨 Shape of category mask:', bool_mask.shape)
        hair_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        hair_mask[bool_mask] = 1  # Hair class index is 1

        return hair_mask
        
