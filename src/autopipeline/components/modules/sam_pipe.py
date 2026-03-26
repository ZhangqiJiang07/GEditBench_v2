import numpy as np
from PIL import Image
from typing import List, Tuple
from ..primitives.visual_consistency import SAMSegmentationMixin
from ..primitives.mask_processor import MaskProcessor
from . import BasePipe, PIPE_REGISTRY
from common_utils.logging_util import get_logger
logger = get_logger()

@PIPE_REGISTRY.register("sam-pipe")
class SAMPipe(SAMSegmentationMixin, MaskProcessor, BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _compute_iou_in_single_bbox(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 1.0
        return intersection / union

    def calc_iou(self, ref_image, edited_image, coords=None):
        if coords is None:
            logger.debug("No coordinates provided for SAM segmentation.")
            return None
        ref_image_np = np.array(ref_image.convert("RGB")) if ref_image.mode != "RGB" else np.array(ref_image)
        edited_image_np = np.array(edited_image.convert("RGB")) if edited_image.mode != "RGB" else np.array(edited_image)

        iou_scores = []
        for _, bbox in enumerate(coords):
            bbox_np = np.array(bbox)
            ref_mask = self.get_best_mask_in_bbox(ref_image_np, bbox_np)
            edited_mask = self.get_best_mask_in_bbox(edited_image_np, bbox_np)
            if ref_mask is None or edited_mask is None:
                logger.debug(f"Could not obtain masks for bbox {bbox}. Skipping IoU computation for this region.")
                continue
            iou = self._compute_iou_in_single_bbox(ref_mask, edited_mask)
            iou_scores.append(iou)
        if len(iou_scores) == 0:
            logger.debug("No valid IoU scores computed. Returning 0.0.")
            return None
        average_iou = float(np.mean(iou_scores))
        return average_iou
    
    def __call__(
        self,
        ref_image: Image.Image,
        edited_image: Image.Image,
        coords: List[Tuple[int, int, int, int]]=None,
        mask_mode: str =None,
        metric: str ='iou',
        **kwargs
    ):
        if metric == 'iou':
            return self.calc_iou(ref_image, edited_image, coords)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        
if __name__ == "__main__":
    sam_pipe = SAMPipe(
        model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml',
        model_path='/path/to/facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt'
    )
    print("SAMPipe initialized successfully!")