import cv2
import numpy as np
from PIL import Image
from . import BasePipe, PIPE_REGISTRY
from common_utils.logging_util import get_logger
logger = get_logger()


@PIPE_REGISTRY.register("hair-consistency-pipe")
class HairConsistencyPipe(BasePipe):
    def color_hist_distance(self, Io, Mo, Ie, Me, bins=32, eps=1e-6):
        """
        Io, Ie: np.ndarray, RGB image, shape (H, W, 3)
        Mo, Me: np.ndarray, binary mask, shape (H, W)
        Lower is more similar
        """

        def masked_lab_hist(image, mask):
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            hist_all = []

            for c in range(3):
                channel = lab[:, :, c][mask > 0]
                hist, _ = np.histogram(
                    channel,
                    bins=bins,
                    range=(0, 256),
                    density=True
                )
                hist_all.append(hist)

            return np.concatenate(hist_all)

        h1 = masked_lab_hist(Io, Mo)
        h2 = masked_lab_hist(Ie, Me)

        # Chi-square distance
        chi2 = 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps))
        return float(chi2)

    def texture_energy_diff(self, Io, Mo, Ie, Me, eps=1e-6):
        """
        Laplacian-based texture energy difference
        Lower is more similar
        """

        def texture_energy(image, mask):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            energy = np.mean(np.abs(lap)[mask > 0])
            return energy

        e1 = texture_energy(Io, Mo)
        e2 = texture_energy(Ie, Me)

        # relative difference
        diff = abs(e1 - e2) / (e1 + e2 + eps)
        return float(diff)

    def high_freq_diff(self, Io, Mo, Ie, Me, eps=1e-6):
        """
        High-frequency difference using Sobel gradients
        Lower is more similar
        """

        def high_freq_energy(image, mask):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

            mag = np.sqrt(gx**2 + gy**2)
            return np.mean(mag[mask > 0])

        hf1 = high_freq_energy(Io, Mo)
        hf2 = high_freq_energy(Ie, Me)

        diff = abs(hf1 - hf2) / (hf1 + hf2 + eps)
        return float(diff)

    def __call__(
        self,
        cropped_ref_human_image: Image.Image,
        cropped_edited_human_image: Image.Image,
        ref_hair_mask: np.ndarray=None,
        edited_hair_mask: np.ndarray=None,
        metric: str = "color_distance",
        **kwargs
    ):
        if ref_hair_mask is None or edited_hair_mask is None:
            logger.debug("Hair mask not found in one of the images.")
            return None
        Io = np.array(cropped_ref_human_image.convert('RGB'))
        Ie = np.array(cropped_edited_human_image.convert('RGB'))
        if "color_distance" in metric:
            bins = kwargs.get('texture_bins', 32)
            return self.color_hist_distance(Io, ref_hair_mask, Ie, edited_hair_mask, bins=bins)
        elif "texture_energy_diff" in metric:
            return self.texture_energy_diff(Io, ref_hair_mask, Ie, edited_hair_mask)
        elif "high_frequency_diff" in metric:
            return self.high_freq_diff(Io, ref_hair_mask, Ie, edited_hair_mask)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

