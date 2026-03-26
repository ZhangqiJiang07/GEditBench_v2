import torch
import numpy as np
from PIL import Image
from typing import Optional
from common_utils.logging_util import get_logger
logger = get_logger()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SSIMMixin:
    def compute(
        self,
        ref_img_tensor: torch.Tensor,
        edited_img_tensor: torch.Tensor,
        mask=None,
        win_size=7,
        win_sigma=1.5,
    ):
        from .ssim import ssim
        ssim_score = ssim(
            ref_img_tensor, edited_img_tensor,
            data_range=255,
            win_size=win_size, win_sigma=win_sigma,
            size_average=True, mask=mask
        )
        if ssim_score.isnan():
            ssim_score = torch.tensor(-1e8)
            logger.debug(f"SSIM score is nan, and set to {ssim_score.item()}")
        return float(ssim_score.cpu().numpy())


class LPIPSMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from lpips import LPIPS
        self.device = kwargs.get('device', DEVICE)
        self.model = LPIPS(net=kwargs.get('net', 'alex')).to(self.device)
        logger.info(f"[LPIPS] model loaded with net: {kwargs.get('net', 'alex')}")

    def compute(self, ref_img_tensor: torch.Tensor, edited_img_tensor: torch.Tensor):
        ref_img_tensor = ref_img_tensor.to(self.device)
        edited_img_tensor = edited_img_tensor.to(self.device)
        factor = 255./2.
        cent = 1.
        ref_img_tensor = (ref_img_tensor / factor - cent)
        edited_img_tensor = (edited_img_tensor / factor - cent)
        lpips_score = self.model.forward(
            ref_img_tensor, edited_img_tensor
        ).cpu()
        return lpips_score.item()


class SAMSegmentationMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .sam2.build_sam import build_sam2
        from .sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device = kwargs.get('device', DEVICE)
        model_cfg = kwargs.get('model_cfg')
        model_path = kwargs.get('model_path')

        self.predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, model_path, device=self.device)
        )
        logger.info("SAM initialized successfully!")

    def get_best_mask_in_bbox(self, image_np: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            try:
                self.predictor.set_image(image_np)
            except Exception as e:
                logger.error(f"Error in setting image for SAM predictor: {e}")
                return None
            masks, scores, _ = self.predictor.predict(
                box=bbox,
                multimask_output=True
            )
            best_mask_index = np.argmax(scores)
            best_mask = masks[best_mask_index]

        return best_mask

    def crop_and_isolate_subject(
        self,
        image_np: np.ndarray,
        bbox: np.ndarray,
        mask: np.ndarray,
        bg_color: tuple = (255, 255, 255),
    ) -> np.ndarray:
        """根据 Mask 提取物体，替换背景为纯白，并按 BBox 裁剪"""
        # 1. 创建全白画布
        white_canvas = np.full_like(image_np, bg_color, dtype=np.uint8)
        
        # 2. 隔离主体 (Mask 为 True 的地方保留原图，False 的地方用白布)
        isolated_image = np.where(mask[..., None], image_np, white_canvas)
        
        # 3. 边界安全处理
        h, w = image_np.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 4. 裁剪
        return isolated_image[y1:y2, x1:x2]

    def extract_object_by_coord(self, image_np: np.ndarray, coord, bg_color=(255, 255, 255)):
        bbox_np = np.array(coord)
        mask = self.get_best_mask_in_bbox(image_np, bbox_np)
        if mask is None:
            logger.warning("Failed to extract mask from SAM.")
            return None

        cropped_isolated = self.crop_and_isolate_subject(image_np, bbox_np, mask, bg_color)
        return cropped_isolated

    
class DepthAnythingv2Mixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.device = kwargs.get('device', DEVICE)
        model_path = kwargs.get('model_path')

        self.model = AutoModelForDepthEstimation.from_pretrained(model_path).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_path)

    def get_depth_map(self, image: Image.Image, resize_to_original: bool =True) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        if resize_to_original:
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
        else:
            prediction = predicted_depth.unsqueeze(1)
        depth_map = prediction.squeeze().cpu().numpy()
        # print(f"🚨 [DEBUG] Depth map shape: {depth_map.shape}")
        return depth_map