
from ..pipelines import PIPELINE_REGISTRY
from common_utils.logging_util import get_logger
from common_utils.project_paths import normalize_benchmark_name
logger = get_logger()

class EvalWorker:
    def __init__(self, pipeline_name):
        self.pipeline_name = pipeline_name
        assert pipeline_name in ["vlm-as-a-judge"], f"Unsupported pipeline: {pipeline_name}"
    
    def __call__(self, item_key, input_dict, runtime, **kwargs):
        pipeline = runtime.get_pipeline()
        res_dict = {}
        if normalize_benchmark_name(kwargs.get("benchmark_name", "")) == "geditv2":
            res_dict["input_dict"] = input_dict
        if input_dict.get("winner", None):
            res_dict["gt_winner"] = input_dict["winner"]
            
        res_dict["winner"], res_dict["raw_responses"] = pipeline(input_dict, **kwargs)
        return item_key, res_dict

class AnnotatorWorker:
    def __init__(self, pipeline_name, edit_task):
        self.edit_task = edit_task.upper()
        self.pipeline_name = pipeline_name
        assert pipeline_name in PIPELINE_REGISTRY.registered_keys(), f"Unsupported pipeline: {pipeline_name}"
    
    def __call__(self, item_key, input_dict, runtime, **kwargs):
        pipeline = runtime.get_pipeline()
        res_dict = {"input_dict": input_dict}

        if self.pipeline_name == "vlm-as-a-judge":
            
            res_dict["winner"], res_dict["raw_responses"] = pipeline(input_dict, **kwargs)
            # logger.info(f"✅ Processing item {item_key}")
            return item_key, res_dict

        elif self.pipeline_name in ['human-centric', 'object-centric']:
            res_dict['output'] = {}
            res_dict['output']['source_image_path'] = input_dict['input_image']
            res_dict['output']['edited_image_path'] = input_dict['edited_images'][0]
            res_dict['output']['instruction'] = input_dict['instruction']

            input_dict.update({"edit_task": self.edit_task})
            scores_dict, raw_responses = pipeline(input_dict, **kwargs)
            for scope, scores in scores_dict.items():
                temp_scores = {}
                for metric_name, score in scores.items():
                    try:
                        temp_scores[metric_name] = float(score)
                    except:
                        logger.info(f"{item_key}: Failed to get scores for [{metric_name}].")
                        temp_scores[metric_name] = None
                res_dict["output"][scope] = temp_scores
            # logger.info(f"✅ Processing item {item_key}。")
            return item_key, res_dict
        else:
            raise NotImplementedError(f"Unsupported pipeline: {self.pipeline_name}")


