import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from ..primitives.visual_consistency import LPIPSMixin
from ..primitives.mask_processor import MaskProcessor
from . import BasePipe, PIPE_REGISTRY


@PIPE_REGISTRY.register("lpips-pipe")
class LPIPSPipe(LPIPSMixin, MaskProcessor, BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc_lpips(self, ref_image, edited_image, mask=None):
        if mask is not None:
            edited_img_filled = np.asarray(edited_image) * mask.transpose(1, 2, 0) + np.asarray(ref_image) * (1 - mask.transpose(1, 2, 0))
        else:
            edited_img_filled = np.asarray(edited_image)
        ref_img_input = torch.from_numpy(np.asarray(ref_image).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32))
        edited_img_input = torch.from_numpy(edited_img_filled.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32))
        lpips_score = self.compute(ref_img_input, edited_img_input)
        if mask is None:
            return lpips_score
        computed_area = np.sum(mask[0])
        img_area = ref_image.size[0] * ref_image.size[1]
        computed_area_lpips = lpips_score / (computed_area / img_area) if computed_area > 0 else lpips_score
        return computed_area_lpips
    
    def __call__(
        self,
        ref_image: Image.Image,
        edited_image: Image.Image,
        coords: List[Tuple[int, int, int, int]]=None,
        mask_mode: str =None,
        metric: str = 'lpips',
        **kwargs
    ):
        img_w, img_h = ref_image.size
        mask = self.make_mask(img_h, img_w, coords, return_format='3d_numpy', mode=mask_mode)
        if metric == 'lpips':
            return self.calc_lpips(ref_image, edited_image, mask)
        else:
            raise ValueError(f"Unsupported metric: {metric}")