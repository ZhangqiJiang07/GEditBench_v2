


from core.registry import Registry

EXPERT_REGISTRY = Registry(name="Experts", enable_regex=False)
from .human_segmenter import HumanSegmentationMixin, HairSegmentationMixin
from .face_analyzer import FaceDetectionMixin

PROMPT_ADAPTER_REGISTRY = Registry("PromptAdapters", enable_regex=False)
from . import prompt_adapters