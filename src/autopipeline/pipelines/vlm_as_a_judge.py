from .base_pipeline import BasePipeline
from . import PIPELINE_REGISTRY



@PIPELINE_REGISTRY.register("vlm-as-a-judge")
class VLMAsAJudgePipeline(BasePipeline):
    required_configs = ['metric_configs']

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.edit_task = kwargs.get('edit_task', None)
        self.metric_to_pipekey, self.pipekey_to_metrics = self.parse_metric_configs(
            self.pipeline_config['metric_configs']
        )
        self.reject_sampling_pipes = self.smart_load_pipes(self.metric_to_pipekey)

    def _prepare_input_dict(self, input_dict, image_max_side):
        valid_input_dict = {}
        try:
            valid_input_dict['instruction'] = input_dict['instruction']
            valid_input_dict['input_image'] = self.parse_image_info(input_dict['input_image'], max_side=image_max_side)
            valid_input_dict['edited_images'] = [
                self.parse_image_info(edit_image, max_side=image_max_side) for edit_image in input_dict['edited_images']
            ]
            if self.edit_task:
                valid_input_dict['edit_task'] = self.edit_task
        except KeyError as e:
            self.logger.error(f"Missing key in input_dict: {e}")
        return valid_input_dict

    def aggregate_results(self, results_dicts):
        # use majority voting to aggregate results from different metrics
        image_a_votes = 0
        image_b_votes = 0
        tie_votes = 0
        for _, result_dict in results_dicts.items():
            winner = result_dict['value']
            if winner == "Image A":
                image_a_votes += 1
            elif winner == "Image B":
                image_b_votes += 1
            elif winner == "Tie":
                tie_votes += 1
        if image_a_votes > image_b_votes and image_a_votes > tie_votes:
            return "Image A"
        elif image_b_votes > image_a_votes and image_b_votes > tie_votes:
            return "Image B"
        elif tie_votes > image_a_votes and tie_votes > image_b_votes:
            return "Tie"
        else:
            return "Failed"

    def __call__(self, input_dict, **kwargs):
        image_max_side = kwargs.get('image_max_side', 1024)
        valid_input_dict = self._prepare_input_dict(input_dict, image_max_side)
        results_dict = {}
        for metric_name, pipe_info in self.reject_sampling_pipes.items():
            results_dict[metric_name] = pipe_info['pipe'](
                valid_input_dict,
                **pipe_info['runtime_params']
            )
        winner = self.aggregate_results(results_dict)
        return winner, results_dict
            
        

        

