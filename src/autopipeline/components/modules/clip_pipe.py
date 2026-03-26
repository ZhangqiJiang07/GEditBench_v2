import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from ..primitives.semantic_consistency import CLIPMixin
from ..primitives.mask_processor import MaskProcessor
from ..primitives.visual_consistency import SAMSegmentationMixin
from . import BasePipe, PIPE_REGISTRY
from common_utils.logging_util import get_logger
logger = get_logger()


@PIPE_REGISTRY.register("clip-pipe")
class CLIPPipe(CLIPMixin, MaskProcessor, BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'sam' in kwargs:
            self.sam_block = SAMSegmentationMixin(**dict(kwargs['sam']))
    
    def calc_emd(self, ref_image, edited_image, bool_mask):
        import ot
        from scipy.spatial.distance import cdist
        ref_img_embeddings = self.get_features(ref_image).cpu().numpy()
        computed_edited_img_embeddings = self.get_features(edited_image, mask=bool_mask).cpu().numpy()
        num_computed_embeds = computed_edited_img_embeddings.shape[0]
        # compute EMD
        weights_orig = np.full(ref_img_embeddings.shape[0], 1/ref_img_embeddings.shape[0], dtype=np.float64)
        weights_edit = np.full(num_computed_embeds, 1/num_computed_embeds, dtype=np.float64)
        cost_matrix = cdist(ref_img_embeddings, computed_edited_img_embeddings, 'sqeuclidean')
        emd_distance = ot.emd2(weights_orig, weights_edit, cost_matrix)
        return emd_distance

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

        interpolation = cv2.INTER_AREA if max_dim > self.img_input_size else cv2.INTER_CUBIC
        resized_img = cv2.resize(
            padded_img, 
            (self.img_input_size, self.img_input_size), 
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
            ref_emb_norm = ref_obj_cls_emb / ref_obj_cls_emb.norm(p=2, dim=-1, keepdim=True)
            edit_emb_norm = edit_obj_cls_emb / edit_obj_cls_emb.norm(p=2, dim=-1, keepdim=True)
            sim_list.append(torch.sum(ref_emb_norm * edit_emb_norm).item())
        return float(np.mean(sim_list))

    def __call__(
        self,
        ref_image: Image.Image,
        edited_image: Image.Image,
        coords: List[Tuple[int, int, int, int]]=None,
        mask_mode: str =None,
        metric: str = 'emd',
        **kwargs
    ):
        if metric in ['emd']:
            img_w, img_h = ref_image.size
            mask = self.make_resized_mask(
                img_h, img_w, coords,
                return_format='2d_numpy', mode=mask_mode,
                target_h=self.img_input_size, target_w=self.img_input_size,
            )
            if mask is not None:
                patch_mask = self.create_patch_mask_from_mask_2d(
                    mask, self.patch_size, threshold=kwargs.get('patch_mask_threshold', 0.1)
                ).flatten()
                if patch_mask.sum() == 0:
                    logger.debug("No valid unedited patches found in the specified regions. Returning large EMD score.")
                    return None
                bool_patch_mask = (patch_mask == 1)
            else:
                bool_patch_mask = None

        if metric == 'emd':
            return self.calc_emd(ref_image, edited_image, bool_patch_mask)
        elif metric == 'sam_clip_cls_sim':
            bg_color = kwargs.get('bg_color', (255, 255, 255))
            return self.calc_object_pad_cls_sim(ref_image, edited_image, coords, bg_color)
        else:
            raise ValueError(f"Unsupported measurement: {metric}")