from PIL import Image
from typing import List, Tuple
import cv2
import numpy as np
from ..primitives.face_analyzer import FaceMeshMixin, FaceIDMixin
from ..constant import FACE_GEOMETRY_LANDMARKS_INDEXES
from . import BasePipe, PIPE_REGISTRY
from common_utils.logging_util import get_logger
logger = get_logger()

def face_crop_with_pad(
    image: Image.Image,
    face_bbox: Tuple[int, int, int, int],
    pad_ratio: float=0.1
) -> np.ndarray:
    image_rgb = np.array(image.convert("RGB"))
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = face_bbox
    w_box = x2 - x1
    h_box = y2 - y1
    # Padding
    pad_w = int(w_box * pad_ratio)
    pad_h = int(h_box * pad_ratio)
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w_img, x2 + pad_w)
    cy2 = min(h_img, y2 + pad_h)
    return image_rgb[cy1:cy2, cx1:cx2]

@PIPE_REGISTRY.register("face-geometry-pipe")
class FaceGeometryPipe(FaceMeshMixin, BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc_L2_distance(self, landmarks1, landmarks2):
        return np.linalg.norm(landmarks1 - landmarks2, axis=1).mean()
    
    def __call__(
        self,
        cropped_ref_human_image: Image.Image,
        cropped_edited_human_image: Image.Image,
        ref_face_bbox: Tuple[int, int, int, int],
        edited_face_bbox: Tuple[int, int, int, int],
        metric: str = "L2_distance",
        **kwargs
    ):
        cropped_ref_face = face_crop_with_pad(
            cropped_ref_human_image, ref_face_bbox,
            pad_ratio=kwargs.get('pad_ratio', 0.1)
        )
        cropped_edited_face = face_crop_with_pad(
            cropped_edited_human_image, edited_face_bbox,
            pad_ratio=kwargs.get('pad_ratio', 0.1)
        )
        ref_landmarks = self.get_face_landmarks_from_cropped_face(cropped_ref_face)
        edited_landmarks = self.get_face_landmarks_from_cropped_face(cropped_edited_face)
        if ref_landmarks is None or edited_landmarks is None:
            logger.debug("No landmarks detected in one of the faces. Returning score of 0.0.")
            return 0.0
        key_ref_landmarks = ref_landmarks[FACE_GEOMETRY_LANDMARKS_INDEXES, :]
        key_edited_landmarks = edited_landmarks[FACE_GEOMETRY_LANDMARKS_INDEXES, :]
        # normalize and align
        normalized_ref_landmarks = self.normalize_landmarks(key_ref_landmarks)
        normalized_edited_landmarks = self.normalize_landmarks(key_edited_landmarks)
        aligned_edited_landmarks = self.procrustes_align(normalized_ref_landmarks, normalized_edited_landmarks)
        # compute L2 distance
        if "L2_distance" in metric:
            return self.calc_L2_distance(normalized_ref_landmarks, aligned_edited_landmarks)
        else:
            raise NotImplementedError(f"Metric {metric} not implemented in FaceGeometryPipe.")

@PIPE_REGISTRY.register("face-texture-pipe")
class FaceTexturePipe(BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def texture_resize(self, face_crop: np.ndarray, texture_size: int) -> np.ndarray:
        return cv2.resize(face_crop, (texture_size, texture_size))
    
    def high_freq(self, face_crop, texture_sigma):
        blur = cv2.GaussianBlur(face_crop, (0, 0), texture_sigma)
        return face_crop.astype(np.float32) - blur.astype(np.float32)
    
    def ab_hist(self, face_crop, texture_bins):
        lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
        a, b = lab[..., 1], lab[..., 2]

        h_a = cv2.calcHist([a], [0], None, [texture_bins], [0, 256])
        h_b = cv2.calcHist([b], [0], None, [texture_bins], [0, 256])

        h = np.concatenate([h_a, h_b], axis=0)
        return h / (np.sum(h) + 1e-6)
    
    def texture_energy(self, face_crop):
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        return np.mean(np.abs(lap))
    
    def __call__(
        self,
        cropped_ref_human_image: Image.Image,
        cropped_edited_human_image: Image.Image,
        ref_face_bbox: Tuple[int, int, int, int],
        edited_face_bbox: Tuple[int, int, int, int],
        metric: str = "high_frequency_diff",
        **kwargs
    ):
        cropped_ref_face = face_crop_with_pad(
            cropped_ref_human_image, ref_face_bbox,
            pad_ratio=kwargs.get('pad_ratio', 0.1)
        )
        cropped_edited_face = face_crop_with_pad(
            cropped_edited_human_image, edited_face_bbox,
            pad_ratio=kwargs.get('pad_ratio', 0.1)
        )

        target_size = kwargs.get('texture_size', 224)
        resized_ref_crop_face = self.texture_resize(cropped_ref_face, target_size)
        resized_edited_crop_face = self.texture_resize(cropped_edited_face, target_size)

        if "high_frequency_diff" in metric:
            sigma = kwargs.get('texture_sigma', 3.0)
            return np.mean(
                np.abs(self.high_freq(resized_ref_crop_face, sigma) - self.high_freq(resized_edited_crop_face, sigma))
            )
        elif "color_similarity" in metric:
            bins = kwargs.get('texture_bins', 32)
            return cv2.compareHist(
                self.ab_hist(resized_ref_crop_face, bins),
                self.ab_hist(resized_edited_crop_face, bins),
                cv2.HISTCMP_INTERSECT
            )
        elif "energy_ratio" in metric:
            e_o = self.texture_energy(resized_ref_crop_face)
            e_e = self.texture_energy(resized_edited_crop_face)
            return min(e_o, e_e) / (max(e_o, e_e) + 1e-6)
        else:
            raise NotImplementedError(f"Metric {metric} not implemented in FaceTexturePipe.")


@PIPE_REGISTRY.register("face-identity-pipe")
class FaceIdentityPipe(FaceIDMixin, BasePipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_cosine_sim(self, feat1, feat2):
        norm_feat1 = np.linalg.norm(feat1)
        norm_feat2 = np.linalg.norm(feat2)
        return np.dot(feat1, feat2) / (norm_feat1 * norm_feat2 + 1e-6)
    
    def _get_best_match_face_embedding_by_bbox(self, subject_crop, subject_face_bbox):
        faces = self.detect_faces(subject_crop)
        if len(faces) == 0:
            logger.debug("[Face Detection]: No faces detected!")
            return None
        max_iou = 0.0
        best_face = None
        for face in faces:
            # face.bbox 是 numpy array [x1, y1, x2, y2]
            iou = self.compute_two_faces_iou(face.bbox, subject_face_bbox)
            if iou > max_iou:
                max_iou = iou
                best_face = face
        if best_face is None or max_iou < 0.3:
            return None
        return best_face.embedding

    def compute_subject_face_consistency(
        self,
        cropped_ref_human_image,
        cropped_edited_human_image,
        ref_face_bbox,
        edited_face_bbox
    ):
        ref_subject_emb = self._get_best_match_face_embedding_by_bbox(cropped_ref_human_image, ref_face_bbox)
        edited_subject_emb = self._get_best_match_face_embedding_by_bbox(cropped_edited_human_image, edited_face_bbox)
        if ref_subject_emb is None or edited_subject_emb is None:
            logger.debug("👻 Unable to get face embedding!")
            return 0.0
        return self._compute_cosine_sim(ref_subject_emb, edited_subject_emb)
    
    def _union_coords_to_subject_bbox(self, coords):
        x1 = min(box[0] for box in coords)
        y1 = min(box[1] for box in coords)
        x2 = max(box[2] for box in coords)
        y2 = max(box[3] for box in coords)
        return (x1, y1, x2, y2)

    def compute_bg_face_consistency(self, ref_image, edited_image, coords):
        """
        计算非编辑区域人脸的一致性
        
        参数:
        img_src, img_edit: numpy array (BGR format, cv2 read)
        coords: [(x1, y1, x2, y2)_1, (x1, y1, x2, y2)_2] 编辑区域
        
        返回:
        avg_score: float (所有背景人脸相似度的平均值，如果没有背景人脸则返回 None)
        """
        subject_bbox = self._union_coords_to_subject_bbox(coords)
        faces_ref = self.detect_faces(ref_image)
        faces_edit = self.detect_faces(edited_image)
        bg_faces_ref = [f for f in faces_ref if not self.is_subject_face(f.bbox, subject_bbox)]
        bg_faces_edit = [f for f in faces_edit if not self.is_subject_face(f.bbox, subject_bbox)]
        if len(bg_faces_ref) == 0:
            logger.debug("No faces in background!")
            return None

        scores = []
        for f_ref in bg_faces_ref:
            best_iou = 0
            best_match = None
            # Use IoU to match the face after editing
            for f_edit in bg_faces_edit:
                iou = self.compute_two_faces_iou(f_ref.bbox, f_edit.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = f_edit
            
            if best_match is not None and best_iou > 0.3:
                sim = self._compute_cosine_sim(f_ref.embedding, best_match.embedding)
                scores.append(sim)
            else:
                scores.append(0.0)
        if not scores:
            return None
        return np.mean(scores)

    def compute_max_match_face_consistency(self, ref_image, edited_image, coords):
        max_match_face_ID_scores = []
        for bbox in coords:
            try:
                cropped_ref_image = ref_image.crop(bbox)
            except:
                logger.debug("Invalid bbox range!")
                continue
            ref_face = self.get_highest_confidence_face(cropped_ref_image)
            faces_edit = self.detect_faces(edited_image)
            max_face_ID_score = 0
            for f_edit in faces_edit:
                score = self._compute_cosine_sim(ref_face.embedding, f_edit.embedding)
                if score > max_face_ID_score:
                    max_face_ID_score = score
            max_match_face_ID_scores.append(max_face_ID_score)
        return np.mean(max_match_face_ID_scores)

    def __call__(
        self,
        ref_image: Image.Image=None,
        edited_image: Image.Image=None,
        coords: List[Tuple[int, int, int, int]]=None,
        cropped_ref_human_image: Image.Image=None,
        cropped_edited_human_image: Image.Image=None,
        ref_face_bbox: Tuple[int, int, int, int]=None,
        edited_face_bbox: Tuple[int, int, int, int]=None,
        metric: str="bg_faceID_sim",
        **kwargs
    ):  
        if metric == 'bg_faceID_sim':
            if coords is None:
                logger.debug("\"subject_bbox\" is needed for computing background faces consistency!")
                return None
            return self.compute_bg_face_consistency(ref_image, edited_image, coords)
        elif metric == 'face_ID_sim':
            if ref_face_bbox is None and edited_face_bbox is None:
                logger.debug("Bounding Box is missing for computing subject face ID similarity!")
                return None
            return self.compute_subject_face_consistency(cropped_ref_human_image, cropped_edited_human_image, ref_face_bbox, edited_face_bbox)
        elif metric == 'max_match_face_ID_sim':
            if coords is None:
                logger.debug("\"subject_bbox\" is needed for computing background faces consistency!")
                return None
            return self.compute_max_match_face_consistency(ref_image, edited_image, coords)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
