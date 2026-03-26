import numpy as np
import torch

class MaskProcessor:
    def make_mask(
        self,
        image_h: int,
        image_w: int,
        coords: list,
        *,
        return_format: str,
        mode: str = None,
    ):
        if mode is None:
            return None
        if return_format not in {"2d_numpy", "3d_numpy", "4d_tensor"}:
            raise ValueError(f"Unsupported return_format: {return_format}")

        if return_format == "2d_numpy": # dino/clip
            shape = (image_h, image_w)
            slicer = lambda m, y1, y2, x1, x2: m[y1:y2, x1:x2]
        elif return_format == "3d_numpy": # lpips
            shape = (3, image_h, image_w)
            slicer = lambda m, y1, y2, x1, x2: m[:, y1:y2, x1:x2]
        else:  # 4d_tensor (ssim)
            shape = (1, 3, image_h, image_w)
            slicer = lambda m, y1, y2, x1, x2: m[..., y1:y2, x1:x2]

        init_val, fill_val = (1, 0) if mode == "outer" else (0, 1)
        mask = np.full(shape, init_val, dtype=np.uint8)

        for x1, y1, x2, y2 in coords:
            slicer(mask, y1, y2, x1, x2)[...] = fill_val

        if return_format == "4d_tensor":
            return torch.from_numpy(mask == 1)

        return mask
    
    def make_resized_mask(
        self,
        image_h: int,
        image_w: int,
        coords: list,
        *,
        return_format: str,
        mode: str = "outer",
        target_h: int,
        target_w: int,
    ):
        resized_coords_list = []
        for (x1, y1, x2, y2) in coords:
            resized_x1 = int(x1 * target_w / image_w)
            resized_y1 = int(y1 * target_h / image_h)
            resized_x2 = int(x2 * target_w / image_w)
            resized_y2 = int(y2 * target_h / image_h)
            resized_coords_list.append((resized_x1, resized_y1, resized_x2, resized_y2))
        return self.make_mask(
            target_h, target_w, resized_coords_list,
            return_format=return_format, mode=mode
        )
        
    def create_patch_mask_from_mask_2d(
        self,
        mask_2d: np.ndarray,
        patch_size: int,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        通过分块投票的方式，从高分辨率图像Mask生成低分辨率的Patch Mask。

        :param mask_2d: 高分辨率的二值Mask (HxW)，值为0或1。
        :param patch_size: Vision Transformer的Patch大小 (例如 14, 16, 32)。
        :param threshold: 阈值 (0.0 to 1.0)。一个块内“未编辑像素(1)”的比例
                        必须大于此阈值，才被认为是未编辑的Patch。
        :return: 低分辨率的Patch Mask (hxw)，值为0或1。
        """
        mask_h, mask_w = mask_2d.shape
        h, w = mask_h // patch_size, mask_w // patch_size
        reshaped_mask = mask_2d.reshape(h, patch_size, w, patch_size)
        patches = reshaped_mask.transpose(0, 2, 1, 3)
        patch_means = patches.mean(axis=(2, 3))
        patch_mask = (patch_means > threshold).astype(np.uint8)
        return patch_mask

