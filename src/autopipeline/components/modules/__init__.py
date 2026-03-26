from core.registry import Registry
from abc import ABC, abstractmethod
from typing import Any, Dict

class BasePipe(ABC):
    def __init__(self, **kwargs):
        # Keep cooperative multiple inheritance chain compatible with mixins
        # that call super().__init__(**kwargs).
        super().__init__()

    @abstractmethod
    def __call__(self, messages) -> str:
        pass

PIPE_REGISTRY = Registry(name="Pipes", enable_regex=False)

from . import clip_pipe
from . import depth_anything_v2_pipe
from . import dino_pipe
from . import face_pipe
from . import hair_pipe
from . import human_body
from . import judge
from . import lpips_pipe
from . import parser_grounder
from . import sam_pipe
from . import ssim_pipe
