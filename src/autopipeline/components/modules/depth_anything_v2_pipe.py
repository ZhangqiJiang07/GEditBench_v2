import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from ..primitives.visual_consistency import DepthAnythingv2Mixin, SSIMMixin
from ..primitives.mask_processor import MaskProcessor
from . import BasePipe, PIPE_REGISTRY
from common_utils.logging_util import get_logger
logger = get_logger()


@PIPE_REGISTRY.register("depth-anything-v2-pipe")
class DepthAnythingv2Pipe(DepthAnythingv2Mixin, SSIMMixin, MaskProcessor, BasePipe):
    def __init__(self, **kwargs):
        # DepthAnythingv2Mixin.__init__(self, **kwargs)
        super().__init__(**kwargs)
    
    def _normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
        depth = depth.astype(np.uint8)
        return depth

    def calc_depth_ssim(
        self,
        ref_image,
        edited_image,
        coords=None,
        mask_mode=None,
        win_size=7,
        win_sigma=1.5,
        resize_depth_maps=True,
    ):
        ref_depth_map = self.get_depth_map(ref_image, resize_to_original=resize_depth_maps)
        edited_depth_map = self.get_depth_map(edited_image, resize_to_original=resize_depth_maps)
        ref_depth_map = self._normalize_depth_map(ref_depth_map)
        edited_depth_map = self._normalize_depth_map(edited_depth_map)
        img_h, img_w = ref_depth_map.shape
        mask = self.make_mask(img_h, img_w, coords, return_format='4d_tensor', mode=mask_mode)
        if mask is not None and mask.shape[1] == 3: # treat depth map as single channel input
            mask = mask[:, 0:1, :, :]
        ref_img_input = torch.from_numpy(ref_depth_map).unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, H, W)
        edited_img_input = torch.from_numpy(edited_depth_map).unsqueeze(0).unsqueeze(0).float()
        return self.compute(ref_img_input, edited_img_input, mask=mask, win_size=win_size, win_sigma=win_sigma)
        
    def __call__(
        self,
        ref_image: Image.Image,
        edited_image: Image.Image,
        coords: List[Tuple[int, int, int, int]]=None,
        mask_mode: str =None,
        metric: str = 'depth_ssim',
        **kwargs
    ):
        resize_depth_maps = kwargs.get('resize_depth_maps', True)
        win_size = kwargs.get('win_size', 7)
        win_sigma = kwargs.get('win_sigma', 1.5)
        if metric == 'depth_ssim':
            return self.calc_depth_ssim(
                ref_image, edited_image,
                coords=coords,
                mask_mode=mask_mode,
                win_size=win_size,
                win_sigma=win_sigma,
                resize_depth_maps=resize_depth_maps,
            )
        else:
            raise ValueError(f"Unsupported measurement: {metric}")
        

if __name__ == "__main__":
    da2_pipe = DepthAnythingv2Pipe(
        model_path='/path/to/Depth-Anything-V2-Large-hf'
    )
    print("DepthAnythingv2Pipe initialized successfully!")