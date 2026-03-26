from .base_pipeline import BasePipeline
from . import PIPELINE_REGISTRY

@PIPELINE_REGISTRY.register("object-centric")
class ObjectCentricPipeline(BasePipeline):
    required_configs = ['parser_grounder_config', 'metric_configs']

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.parser_grounder = self._load_parser_grounder_module(
            self.pipeline_config['parser_grounder_config']
        )

        self.metric_to_pipekey, self.pipekey_to_metrics = self.parse_metric_configs(
            self.pipeline_config['metric_configs']
        )
        self.reject_sampling_pipes = self.smart_load_pipes(self.metric_to_pipekey)

    def _prepare_input_dict(self, input_dict, image_max_side):
        valid_input_dict = {}
        try:
            valid_input_dict['instruction'] = input_dict['instruction']
            valid_input_dict['input_image'] = self.parse_image_info(input_dict['input_image'], max_side=image_max_side)
            valid_input_dict['edited_image'] = self.parse_image_info(input_dict['edited_images'][0], max_side=image_max_side)
            if valid_input_dict['edited_image'] != valid_input_dict['input_image']:
                valid_input_dict['edited_image'] = valid_input_dict['edited_image'].resize(valid_input_dict['input_image'].size)
            valid_input_dict['edit_task'] = input_dict['edit_task']
        except KeyError as e:
            self.logger.error(f"Missing key in input_dict: {e}")
        return valid_input_dict

    def __call__(self, input_dict, **kwargs):
        image_max_side = kwargs.get('image_max_side', 2048)
        valid_input_dict = self._prepare_input_dict(input_dict, image_max_side)

        # Step I: Parser & Grounder
        objects_dict, coords = self.parser_grounder(valid_input_dict, **kwargs)
        if objects_dict is None:
            return None, None
        objects_dict['edited_area_ratio'] = self.compute_edited_area_ratio(
            valid_input_dict['input_image'].size, coords, valid_input_dict['edit_task']
        )

        # Step II: Region-specific Metrics Computation
        scores_dict = {}
        for metric_name, pipe_info in self.reject_sampling_pipes.items():
            scores_dict.setdefault(pipe_info['scope'], {})
            score = pipe_info['pipe'](
                ref_image=valid_input_dict['input_image'],
                edited_image=valid_input_dict['edited_image'],
                coords=coords,
                mask_mode=pipe_info['mask_mode'],
                metric=metric_name,
                **pipe_info['runtime_params'],
            )
            scores_dict[pipe_info['scope']][metric_name] = score
            self.logger.debug(f"Computed [{metric_name}] score: {score} for scope [{pipe_info['scope']}]")
        
        return scores_dict, objects_dict

