import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from ..primitives.visual_consistency import SSIMMixin
from ..primitives.mask_processor import MaskProcessor
from . import BasePipe, PIPE_REGISTRY

@PIPE_REGISTRY.register("ssim-pipe")
class SSIMPipe(SSIMMixin, MaskProcessor, BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def calc_RGB_channel_ssim(self, ref_image, edited_image, mask=None, win_size=7, win_sigma=1.5):
        ref_img_input = torch.from_numpy(
            np.asarray(ref_image).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        )
        edited_img_input = torch.from_numpy(
            np.asarray(edited_image).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        )
        return self.compute(ref_img_input, edited_img_input, mask=mask, win_size=win_size, win_sigma=win_sigma)
    
    def _get_L_channel_numpy(self, image):
        arr_img_rgb = np.asarray(image)
        arr_img_L = cv2.cvtColor(arr_img_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        return arr_img_L

    def calc_L_channel_ssim(self, ref_image, edited_image, mask=None, win_size=7, win_sigma=1.5):
        from skimage.exposure import match_histograms
        if mask is not None and mask.shape[1] == 3:
            mask = mask[:, 0:1, :, :]
        ref_L_numpy = self._get_L_channel_numpy(ref_image)
        edited_L_numpy = self._get_L_channel_numpy(edited_image)
        try:
            matched_L_numpy = match_histograms(edited_L_numpy, ref_L_numpy, channel_axis=None)
        except TypeError:
            matched_L_numpy = match_histograms(edited_L_numpy, ref_L_numpy, multichannel=False)
        matched_L_numpy = np.clip(matched_L_numpy, 0, 255).astype(np.uint8)
        ref_img_input = torch.from_numpy(ref_L_numpy).unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, H, W)
        edited_img_input = torch.from_numpy(matched_L_numpy).unsqueeze(0).unsqueeze(0).float()
        return self.compute(ref_img_input, edited_img_input, mask=mask, win_size=win_size, win_sigma=win_sigma)

    def __call__(
        self,
        ref_image: Image.Image,
        edited_image: Image.Image,
        coords: List[Tuple[int, int, int, int]]=None,
        mask_mode: str =None,
        metric: str = 'ssim',
        **kwargs
    ):
        win_size = kwargs.get('win_size', 7)
        win_sigma = kwargs.get('win_sigma', 1.5)
        img_w, img_h = ref_image.size
        mask = self.make_mask(img_h, img_w, coords, return_format='4d_tensor', mode=mask_mode)
        if metric == 'L_channel_ssim':
            return self.calc_L_channel_ssim(ref_image, edited_image, mask=mask, win_size=win_size, win_sigma=win_sigma)
        elif metric == 'ssim':
            return self.calc_RGB_channel_ssim(ref_image, edited_image, mask=mask, win_size=win_size, win_sigma=win_sigma)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
