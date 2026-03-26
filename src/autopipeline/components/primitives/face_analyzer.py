import cv2
# import insightface
import numpy as np
import mediapipe as mp
from PIL import Image
from typing import Tuple
from insightface.app import FaceAnalysis
from . import EXPERT_REGISTRY

@EXPERT_REGISTRY.register("face-detector")
class FaceDetectionMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        min_conf = kwargs.get('min_detection_confidence', 0.5)
        self.mp_face_detection = mp.solutions.face_detection
        # Model 1: 适合大多数情况 (全身、半身、多人、小脸) - 作为主力
        self.detector_long = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_conf
        )
        # Model 0: 适合大脸特写 - 作为补充
        self.detector_short = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_conf
        )

    def detect_faces_adaptive(self, image_rgb: np.ndarray):
        """尝试使用两种模型检测人脸，返回检测结果"""
        results = self.detector_long.process(image_rgb)
        if not results.detections:
            results = self.detector_short.process(image_rgb)
        return results
    
    def get_first_face_bounding_box(self, image: Image.Image) -> Tuple[int, int, int, int]:
        image_rgb = np.array(image.convert("RGB"))
        h_img, w_img = image_rgb.shape[:2]

        det_results = self.detect_faces_adaptive(image_rgb)

        if not det_results or not det_results.detections:
            return None
        # get the first detected face
        target_face = det_results.detections[0]

        bboxC = target_face.location_data.relative_bounding_box
        x1 = int(bboxC.xmin * w_img)
        y1 = int(bboxC.ymin * h_img)
        w_box = int(bboxC.width * w_img)
        h_box = int(bboxC.height * h_img)
        x2 = x1 + w_box
        y2 = y1 + h_box
        # check
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(w_img, x2)
        cy2 = min(h_img, y2)

        return (cx1, cy1, cx2, cy2)

class FaceMeshMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=kwargs.get('min_detection_confidence', 0.5)
        )

    def get_face_landmarks_from_cropped_face(self, face_crop: np.ndarray):
        h_crop, w_crop = face_crop.shape[:2]

        if h_crop < 10 or w_crop < 10:
            return None
        mesh_results = self.mesh.process(face_crop)
        if not mesh_results.multi_face_landmarks:
            return None
        
        landmarks_global = []
        face_landmarks = mesh_results.multi_face_landmarks[0]
        
        for lm in face_landmarks.landmark:
            lx_crop = lm.x * w_crop
            ly_crop = lm.y * h_crop
            landmarks_global.append([lx_crop, ly_crop])
            
        return np.array(landmarks_global)
    
    def normalize_landmarks(self, L: np.ndarray) -> np.ndarray:
        """
        Remove translation and scale.
        L: (N, 2)
        """
        L = L - np.mean(L, axis=0, keepdims=True)
        scale = np.linalg.norm(L)
        if scale < 1e-6:
            return L
        return L / scale
    
    def procrustes_align(
        self,
        L_ref: np.ndarray,
        L: np.ndarray
    ) -> np.ndarray:
        """
        Align L to L_ref using similarity (rotation only after normalization).
        """
        # Both should already be normalized
        H = L.T @ L_ref
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        return L @ R


class FaceIDMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化 InsightFace，默认包含检测(det)和识别(rec)模型
        # providers: ['CUDAExecutionProvider'] or ['CPUExecutionProvider']
        self.device = kwargs.get('device', 'cpu')
        provider = 'CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider'
        self.app = FaceAnalysis(
            name=kwargs.get('face_ana_name', 'buffalo_l'),
            root=kwargs.get('model_root', "~/.insightface"),
            providers=[provider]
        )
        self.app.prepare(ctx_id=0, det_size=kwargs.get('detection_size', (640, 640)))

    def is_subject_face(self, face_bbox, sub_bbox):
        fx1, fy1, fx2, fy2 = face_bbox
        sx1, sy1, sx2, sy2 = sub_bbox
        
        # 计算人脸中心点
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        
        # 简单判断：如果人脸中心点在 subject_bbox 内，就认为是主体
        if sx1 < cx < sx2 and sy1 < cy < sy2:
            return True
        return False
    
    def compute_two_faces_iou(self, boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def detect_faces(self, image: Image.Image):
        # convert RGB to BGR
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.app.get(img_bgr)
    
    def get_highest_confidence_face(self, image: Image.Image):
        """
        检测人脸并返回置信度最高的那一个
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
            
        # find best face according to `det_score`
        best_face = max(faces, key=lambda f: f.det_score)
        return best_face