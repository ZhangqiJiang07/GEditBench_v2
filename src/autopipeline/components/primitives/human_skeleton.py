import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


class HumanSkeletonMixin:
    def __init__(self, **kwargs):
        """
        初始化 MediaPipe Pose 模型
        :param static_mode: True 用于单张图片处理（精度更高），False 用于视频流
        :param model_complexity: 0, 1, 2 (2 为最高精度，但在 CPU 上较慢，推荐用于评估任务)
        """
        super().__init__(**kwargs)
        self.mp_pose = mp.solutions.pose        
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=kwargs.get("static_mode", True),
            model_complexity=kwargs.get("model_complexity", 0),
            enable_segmentation=False,
            min_detection_confidence=kwargs.get("min_detection_confidence", 0.5)
        )

    def get_skeleton_np(self, image: Image.Image):
        image_rgb = np.array(image.convert('RGB'))
        results = self.pose_detector.process(image_rgb)
        if not results.pose_landmarks:
            return None
        landmarks_list = []
        for landmark in results.pose_landmarks.landmark:
            landmarks_list.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        landmarks_np = np.array(landmarks_list)
        return landmarks_np


    
