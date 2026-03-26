from core.registry import Registry



PIPELINE_REGISTRY = Registry(name="Pipeline", enable_regex=False)

from . import vlm_as_a_judge
from . import object_centric
from . import human_centric


class PipelineLoader:
    def __init__(self, pipeline_config):
        self.pipeline_config = pipeline_config
    
    def load(self):
        pipeline_name = self.pipeline_config.get("name")
        if pipeline_name not in PIPELINE_REGISTRY.registered_keys():
            raise ValueError(f"Unsupported pipeline: {pipeline_name}")
        pipeline_class = PIPELINE_REGISTRY.get(pipeline_name)

        return pipeline_class(**self.pipeline_config)