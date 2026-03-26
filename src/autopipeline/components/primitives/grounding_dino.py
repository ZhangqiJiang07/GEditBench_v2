import torch
from PIL import Image
from typing import List
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDINOMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = kwargs.get(
            'model_path',
            'GroundingDINO/GroundingDINO-SwinB'
        )
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)

    def object_detection(self, image: Image.Image, text_labels: List[str], box_threshold=0.25, text_threshold=0.25):
        inputs = self.processor(images=image, text=[text_labels], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        return results[0]  # Return the first (and only) result
    
    def get_bounding_boxes(self, image: Image.Image, text_labels: List[str], box_threshold=0.25, text_threshold=0.25):
        detection_results = self.object_detection(image, text_labels, box_threshold, text_threshold)
        bboxes_info = []
        for box, score, label in zip(detection_results["boxes"], detection_results["scores"], detection_results["text_labels"]):
            box = [round(coord) for coord in box.tolist()]
            bboxes_info.append({
                "box": box,
                "score": score,
                "label": label,
            })
        return bboxes_info