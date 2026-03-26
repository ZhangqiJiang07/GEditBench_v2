import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple
import torch.nn.functional as F
from ..primitives.mask_processor import MaskProcessor
from ..primitives.semantic_consistency import DINOv3Mixin
from ..primitives.human_skeleton import HumanSkeletonMixin
from ..constant import SKELETON_CORE_12_INDICES
from . import BasePipe, PIPE_REGISTRY
from common_utils.logging_util import get_logger
logger = get_logger()

def pad_face_bbox(
    face_bbox: Tuple[int, int, int, int],
    pad_ratio: float=0.3,
    image_h: int = None,
    image_w: int = None
) -> Tuple[int, int, int, int]:
    if face_bbox is None:
        return None
    x1, y1, x2, y2 = face_bbox
    w_box = x2 - x1
    h_box = y2 - y1
    # Padding
    pad_w = int(w_box * pad_ratio)
    pad_h = int(h_box * pad_ratio)
    return (
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(image_w, x2 + pad_w),
        min(image_h, y2 + pad_h)
    )

def get_cleaned_resized_body_mask(
    body_mask,
    hair_mask=None,
    face_bbox=None,
    target_h=None,
    target_w=None,
) -> np.ndarray:
    # Exclude hair and face regions from body mask
    body_mask_cleaned = body_mask.copy()
    # Exclude hair
    if hair_mask is not None:
        body_mask_cleaned[hair_mask > 0] = 0
    # Exclude face
    if face_bbox is not None:
        x1, y1, x2, y2 = map(int, face_bbox)
        # 增加边界检查，防止 bbox 超出图像范围报错
        h, w = body_mask_cleaned.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        body_mask_cleaned[y1:y2, x1:x2] = 0
    # Resize for DINO
    if target_h is not None and target_w is not None:
        body_mask_cleaned = cv2.resize(
            body_mask_cleaned,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )
    return body_mask_cleaned


@PIPE_REGISTRY.register("body-pose-and-shape-pipe")
class BodyPoseAndShapePipe(HumanSkeletonMixin, BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("[Body Pose and Shape] Pipe is loaded!")

    def calc_skeleton_aligned_shape_iou(
        self,
        ref_body_mask: np.ndarray,
        edited_body_mask: np.ndarray,
        ref_lm: np.ndarray,
        edited_lm: np.ndarray,
        ref_size: tuple,
        edited_size: tuple,
        target_size: tuple=(512, 512)
    ) -> float:
        def get_align_params(lm, img_w, img_h):
            # 转绝对坐标
            pts = lm[:, :2] * np.array([img_w, img_h])
            core_pts = pts[SKELETON_CORE_12_INDICES]
            vis = lm[SKELETON_CORE_12_INDICES, 3]
            valid = core_pts[vis > 0.5]
            if len(valid) < 3:
                return None, None # 无法对齐
            
            # 计算质心 (Center)
            center = np.mean(valid, axis=0)
            # 计算尺度 (Scale) - 这里的尺度定义为到质心的平均距离
            scale = np.mean(np.linalg.norm(valid - center, axis=1))
            return center, scale
        c_ref, s_ref = get_align_params(ref_lm, ref_size[0], ref_size[1])
        c_edit, s_edit = get_align_params(edited_lm, edited_size[0], edited_size[1])

        if c_ref is None or c_edit is None or s_ref < 1e-6 or s_edit < 1e-6:
            logger.debug("Invalid skeleton, computing IoU directly!")
            intersection = np.logical_and(ref_body_mask, edited_body_mask).sum()
            union = np.logical_or(ref_body_mask, edited_body_mask).sum()
            return intersection / (union + 1e-6)
        
        # --- 3. 构建仿射变换矩阵 ---
        # 目标：将人物移动到 target_size 的中心，并将骨架尺度缩放到固定值 (例如画布的 1/3)
        target_cx, target_cy = target_size[0] // 2, target_size[1] // 2
        target_scale_ref = min(target_size) * 0.35 # 目标骨架半径

        def get_affine_matrix(center, scale):
            # 1. 平移: center -> (0,0)
            T1 = np.array([[1, 0, -center[0]], 
                        [0, 1, -center[1]], 
                        [0, 0, 1]])
            # 2. 缩放: scale -> target_scale_ref
            s_factor = target_scale_ref / scale
            S = np.array([[s_factor, 0, 0], 
                        [0, s_factor, 0], 
                        [0, 0, 1]])
            # 3. 平移: (0,0) -> target_center
            T2 = np.array([[1, 0, target_cx], 
                        [0, 1, target_cy], 
                        [0, 0, 1]])
            
            # 组合矩阵 M = T2 * S * T1
            M_combined = T2 @ S @ T1
            return M_combined[:2, :] # 取前两行用于 cv2.warpAffine
        
        M_src = get_affine_matrix(c_ref, s_ref)
        M_edit = get_affine_matrix(c_edit, s_edit)
        warped_src = cv2.warpAffine(ref_body_mask.astype(np.uint8), M_src, target_size, flags=cv2.INTER_NEAREST)
        warped_edit = cv2.warpAffine(edited_body_mask.astype(np.uint8), M_edit, target_size, flags=cv2.INTER_NEAREST)
        # compute IoU
        warped_src = (warped_src > 0)
        warped_edit = (warped_edit > 0)

        intersection = np.logical_and(warped_src, warped_edit).sum()
        union = np.logical_or(warped_src, warped_edit).sum()
        iou = intersection / (union + 1e-6)
        return iou
    
    def calc_pose_position_error(
        self,
        ref_lm: np.ndarray,
        edited_lm: np.ndarray,
        ref_size: tuple,
        edited_size: tuple
    ) -> float:
        # 1. Preprocess skeleton matrix 
        pts_src_all = ref_lm[:, :2] * np.array(ref_size)
        pts_edit_all = edited_lm[:, :2] * np.array(edited_size)
        # select core 12 points
        p_src = pts_src_all[SKELETON_CORE_12_INDICES]
        p_edit = pts_edit_all[SKELETON_CORE_12_INDICES]
        # visiable?
        vis_src = ref_lm[SKELETON_CORE_12_INDICES, 3]
        vis_edit = edited_lm[SKELETON_CORE_12_INDICES, 3]
        valid_mask = (vis_src > 0.5) & (vis_edit > 0.5)
        if np.sum(valid_mask) < 3:
            logger.debug("Insufficient effective key points to compute.")
            return None
        P = p_src[valid_mask]   # Source Points (N, 2)
        Q = p_edit[valid_mask]  # Edit Points (N, 2)

        # 2. Procrustes Analysis
        # (a) Translation Alignment
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        # (b) Scale Normalization
        scale_P = np.sqrt(np.sum(P_centered ** 2) / len(P))
        scale_Q = np.sqrt(np.sum(Q_centered ** 2) / len(Q))
        if scale_P < 1e-6 or scale_Q < 1e-6:
            return None
        P_norm = P_centered / scale_P
        Q_norm = Q_centered / scale_Q

        # 3. Compute Distance
        distances = np.linalg.norm(P_norm - Q_norm, axis=1)
        # MPJPE (Mean Per Joint Position Error)
        mean_error = np.mean(distances)
        return mean_error # smaller is better

    def __call__(
        self,
        cropped_ref_human_image: Image.Image,
        cropped_edited_human_image: Image.Image,
        ref_face_bbox: Tuple[int, int, int, int]=None,
        edited_face_bbox: Tuple[int, int, int, int]=None,
        ref_hair_mask: np.ndarray=None,
        edited_hair_mask: np.ndarray=None,
        ref_body_mask: np.ndarray=None,
        edited_body_mask: np.ndarray=None,
        metric: str = "body_shape_iou",
        **kwargs
    ):
        if ref_body_mask is None or edited_body_mask is None:
            logger.debug("Body mask not found when computing [Body Shape and Pose] consistency.")
            return None
        # get skeleton
        ref_lm = self.get_skeleton_np(cropped_ref_human_image)
        edited_lm = self.get_skeleton_np(cropped_edited_human_image)
        if ref_lm is None or edited_lm is None:
            logger.debug("🚨 [Human Skeleton] is not exist!")
            return None
        if metric == 'body_shape_iou':
            # clean body masks by removing hair and face regions
            pad_face_bbox_pad_ratio = kwargs.get('face_bbox_pad_ratio', 0.3)
            image_w, image_h = cropped_ref_human_image.size
            ref_body_mask_cleaned = get_cleaned_resized_body_mask(
                ref_body_mask,
                hair_mask=ref_hair_mask,
                face_bbox=pad_face_bbox(ref_face_bbox, pad_ratio=pad_face_bbox_pad_ratio, image_h=image_h, image_w=image_w)
            )
            edited_body_mask_cleaned = get_cleaned_resized_body_mask(
                edited_body_mask,
                hair_mask=edited_hair_mask,
                face_bbox=pad_face_bbox(edited_face_bbox, pad_ratio=pad_face_bbox_pad_ratio, image_h=image_h, image_w=image_w)
            )
            return self.calc_skeleton_aligned_shape_iou(
                ref_body_mask_cleaned, edited_body_mask_cleaned,
                ref_lm=ref_lm, edited_lm=edited_lm,
                ref_size=cropped_ref_human_image.size,
                edited_size=cropped_edited_human_image.size,
                target_size=kwargs.get('target_canvas_size', (512, 512))
            )
        elif metric == 'body_pose_position_error':
            return self.calc_pose_position_error(
                ref_lm=ref_lm, edited_lm=edited_lm,
                ref_size=cropped_ref_human_image.size,
                edited_size=cropped_edited_human_image.size,
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")


@PIPE_REGISTRY.register("body-appearance-pipe")
class BodyAppearancePipe(DINOv3Mixin, MaskProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc_dino_aggregation_cosine_similarity(
        self,
        cropped_ref_human_image,
        cropped_edited_human_image,
        ref_bool_mask,
        edited_bool_mask
    ):
        ref_selected_features = self.get_features(cropped_ref_human_image, mask=ref_bool_mask).cpu()
        edited_selected_features = self.get_features(cropped_edited_human_image, mask=edited_bool_mask).cpu()
        ref_avg_feature = torch.mean(ref_selected_features, dim=0)
        edited_avg_feature = torch.mean(edited_selected_features, dim=0)
        # normalization
        ref_avg_norm_feature = F.normalize(ref_avg_feature, p=2, dim=0)
        edited_avg_norm_feature = F.normalize(edited_avg_feature, p=2, dim=0)
        # compute cosine similarity
        score = torch.dot(ref_avg_norm_feature, edited_avg_norm_feature).item()
        return max(0.0, score) # higher the better

    def __call__(
        self,
        cropped_ref_human_image: Image.Image,
        cropped_edited_human_image: Image.Image,
        ref_face_bbox: Tuple[int, int, int, int]=None,
        edited_face_bbox: Tuple[int, int, int, int]=None,
        ref_hair_mask: np.ndarray=None,
        edited_hair_mask: np.ndarray=None,
        ref_body_mask: np.ndarray=None,
        edited_body_mask: np.ndarray=None,
        metric: str = "body_appearance_dino_cosine_sim",
        **kwargs
    ):
        if ref_body_mask is None or edited_body_mask is None:
            logger.debug("Body mask not found!")
            return None
        pad_face_bbox_pad_ratio = kwargs.get('face_bbox_pad_ratio', 0.3)
        image_w, image_h = cropped_ref_human_image.size
        ref_body_mask_cleaned = get_cleaned_resized_body_mask(
            ref_body_mask,
            hair_mask=ref_hair_mask,
            face_bbox=pad_face_bbox(ref_face_bbox, pad_ratio=pad_face_bbox_pad_ratio, image_h=image_h, image_w=image_w),
            target_h=self.input_image_size, target_w=self.input_image_size
        )
        edited_body_mask_cleaned = get_cleaned_resized_body_mask(
            edited_body_mask,
            hair_mask=edited_hair_mask,
            face_bbox=pad_face_bbox(edited_face_bbox, pad_ratio=pad_face_bbox_pad_ratio, image_h=image_h, image_w=image_w),
            target_h=self.input_image_size, target_w=self.input_image_size
        )
        if metric == 'body_appearance_dino_cosine_sim':
            ref_patch_mask = self.create_patch_mask_from_mask_2d(
                ref_body_mask_cleaned, self.patch_size, threshold=kwargs.get('patch_mask_threshold', 0.1)
            ).flatten()
            edited_patch_mask = self.create_patch_mask_from_mask_2d(
                edited_body_mask_cleaned, self.patch_size, threshold=kwargs.get('patch_mask_threshold', 0.1)
            ).flatten()
            if ref_patch_mask.sum() == 0 or edited_patch_mask.sum() == 0:
                logger.debug("No valid body patches found in the specified regions. Returning score of None.")
                return None
            ref_bool_mask = (ref_patch_mask == 1)
            edited_bool_mask = (edited_patch_mask == 1)
            return self.calc_dino_aggregation_cosine_similarity(
                cropped_ref_human_image, cropped_edited_human_image,
                ref_bool_mask, edited_bool_mask
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")
