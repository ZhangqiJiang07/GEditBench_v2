import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import torch.nn.functional as F
from ..primitives.semantic_consistency import DINOv3Mixin
from ..primitives.mask_processor import MaskProcessor
from ..primitives.visual_consistency import SAMSegmentationMixin
from . import BasePipe, PIPE_REGISTRY
from common_utils.logging_util import get_logger
logger = get_logger()

@PIPE_REGISTRY.register("dino-v3-pipe")
class DINOv3Pipe(DINOv3Mixin, MaskProcessor, BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'sam' in kwargs:
            self.sam_block = SAMSegmentationMixin(**dict(kwargs['sam']))
    
    def _calc_self_sim_matrix(self, features: torch.Tensor) -> torch.Tensor:
        masked_features = F.normalize(features, dim=-1, p=2)
        self_sim_matrix = torch.matmul(masked_features, masked_features.t())
        return self_sim_matrix
    
    def calc_structure_similarity(self, ref_image, edited_image, bool_mask):
        ref_sim_matrix = self._calc_self_sim_matrix(self.get_features(ref_image, mask=bool_mask).cpu())
        edited_sim_matrix = self._calc_self_sim_matrix(self.get_features(edited_image, mask=bool_mask).cpu())
        loss = F.mse_loss(ref_sim_matrix, edited_sim_matrix)
        score = 1.0 / (1.0 + loss.item() * 100.0)
        return score

    def _pad_image_to_target_size(self, cropped_image: np.ndarray, bg_color=(255, 255, 255)):
        h, w = cropped_image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Cropped image has 0 height or width.")

        max_dim = max(h, w)
        top = (max_dim - h) // 2
        bottom = max_dim - h - top
        left = (max_dim - w) // 2
        right = max_dim - w - left

        padded_img = cv2.copyMakeBorder(
            cropped_image,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=bg_color
        )

        interpolation = cv2.INTER_AREA if max_dim > self.input_image_size else cv2.INTER_CUBIC
        resized_img = cv2.resize(
            padded_img, 
            (self.input_image_size, self.input_image_size), 
            interpolation=interpolation
        )
        return resized_img

    def calc_object_pad_cls_sim(self, ref_image, edited_image, coords, bg_color=(255, 255, 255)):
        if len(coords) % 2 != 0:
            return None
        ref_image_np = np.array(ref_image.convert("RGB")) if ref_image.mode != "RGB" else np.array(ref_image)
        edited_image_np = np.array(edited_image.convert("RGB")) if edited_image.mode != "RGB" else np.array(edited_image)
        ref_image_object_coords, edit_image_object_coords = coords[:len(coords)//2], coords[len(coords)//2:]

        sim_list = []
        for ref_obj_coord, edit_obj_coord in zip(ref_image_object_coords, edit_image_object_coords):
            ref_obj_cropped_isolated = self.sam_block.extract_object_by_coord(ref_image_np, ref_obj_coord, bg_color)
            edit_obj_cropped_isolated = self.sam_block.extract_object_by_coord(edited_image_np, edit_obj_coord, bg_color)
            if (ref_obj_cropped_isolated.size == 0) or (edit_obj_cropped_isolated.size == 0):
                logger.warning("Cropped image is empty. Check BBox coordinates.")
                return None

            ref_obj_pad = self._pad_image_to_target_size(ref_obj_cropped_isolated, bg_color)
            edit_obj_pad = self._pad_image_to_target_size(edit_obj_cropped_isolated, bg_color)
            ref_obj_cls_emb = self.get_cls_feature(ref_obj_pad)
            edit_obj_cls_emb = self.get_cls_feature(edit_obj_pad)
            sim_list.append(F.cosine_similarity(ref_obj_cls_emb, edit_obj_cls_emb, dim=0).item())
        return float(np.mean(sim_list))
    
    def __call__(
        self,
        ref_image: Image.Image,
        edited_image: Image.Image,
        coords: List[Tuple[int, int, int, int]]=None,
        mask_mode: str =None,
        metric: str = 'dinov3_structure_similarity',
        **kwargs
    ):
        if metric in ['dinov3_structure_similarity']:
            img_w, img_h = ref_image.size
            mask = self.make_resized_mask(
                img_h, img_w, coords,
                return_format='2d_numpy', mode=mask_mode,
                target_h=self.input_image_size, target_w=self.input_image_size,
            )
            if mask is not None:
                patch_mask = self.create_patch_mask_from_mask_2d(
                    mask, self.patch_size, threshold=kwargs.get('patch_mask_threshold', 0.1)
                ).flatten()
                if patch_mask.sum() == 0:
                    logger.debug("No valid unedited patches found in the specified regions. Returning score of 0.0.")
                    return 0.0
                bool_patch_mask = (patch_mask == 1)
            else:
                bool_patch_mask = None
        
        if metric == 'dinov3_structure_similarity':
            return self.calc_structure_similarity(ref_image, edited_image, bool_patch_mask)
        elif metric == 'sam_dino_cls_sim':
            bg_color = kwargs.get('bg_color', (255, 255, 255))
            return self.calc_object_pad_cls_sim(ref_image, edited_image, coords, bg_color)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
