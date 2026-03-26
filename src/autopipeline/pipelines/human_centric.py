from PIL import Image
from typing import List, Tuple, Dict
from .base_pipeline import BasePipeline
from . import PIPELINE_REGISTRY

@PIPELINE_REGISTRY.register("human-centric")
class HumanCentricPipeline(BasePipeline):
    required_configs = ['parser_grounder_config', 'metric_configs', 'expert_configs']
    attributes_to_metrics_map = {
        "Face ID": ['face_ID_sim'],
        "Face Geometry": ['face_geometry_L2_distance'],
        "Face Texture": ['face_texture_high_frequency_diff', 'face_texture_color_similarity', 'face_texture_energy_ratio'],
        "Hair Appearance": ['hair_color_distance', 'hair_texture_energy_diff', 'hair_high_frequency_diff'],
        "Body Pose&Shape": ['body_shape_iou', 'body_pose_position_error'],
        "Body Appearance": ['body_appearance_dino_cosine_sim']
    }
    experts_to_regions_map = {
        "face-detector": "face",
        "human-segmenter": "body",
        "hair-segmenter": "hair",
    }

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.parser_grounder = self._load_parser_grounder_module(
            self.pipeline_config['parser_grounder_config']
        )

        self.experts = self._load_expert_module(
            self.pipeline_config['expert_configs'], self.pipeline_config['metric_configs']
        )

        self.metric_to_pipekey, self.pipekey_to_metrics = self.parse_metric_configs(
            self.pipeline_config['metric_configs']
        )
        self.reject_sampling_pipes = self.smart_load_pipes(self.metric_to_pipekey)

        self.measure_scopes = set(info['scope'] for info in self.metric_to_pipekey.values())
        self.edit_area_metrics, self.unedit_area_metrics = [], []
        for metric_name, metric_cfg in dict(self.pipeline_config['metric_configs']).items():
            if metric_cfg['scope'] == 'unedit_area':
                self.unedit_area_metrics.append(metric_name)
            elif metric_cfg['scope'] == 'edit_area':
                self.edit_area_metrics.append(metric_name)
            else:
                raise ValueError(f"Invalid region for {metric_name}")
            
    def _crop_human_from_union_bbox(self, image: Image.Image, coords: List[Tuple[int, int, int, int]]) -> Image.Image:
        # compute union bbox
        if not coords:
            return None
        x1 = min(box[0] for box in coords)
        y1 = min(box[1] for box in coords)
        x2 = max(box[2] for box in coords)
        y2 = max(box[3] for box in coords)
        # check
        u_x1 = max(0, x1)
        u_y1 = max(0, y1)
        u_x2 = min(image.width, x2)
        u_y2 = min(image.height, y2)
        if u_x1 >= u_x2 or u_y1 >= u_y2:
            return None
        # crop human region
        union_coords = (u_x1, u_y1, u_x2, u_y2)
        human_crop = image.crop(union_coords)
        return human_crop, union_coords
    
    def _prepare_edit_area_metrics_inputs(
        self,
        ref_human_crop: Image.Image,
        edited_human_crop: Image.Image
    ) -> Dict:
        edit_area_input_dict = {
            "cropped_ref_human_image": ref_human_crop,
            "cropped_edited_human_image": edited_human_crop,
            "ref_face_bbox": None,
            "edited_face_bbox": None,
            "ref_hair_mask": None,
            "edited_hair_mask": None,
            "ref_body_mask": None,
            "edited_body_mask": None,
        }
        edit_area_input_dict['ref_face_bbox'] = self.experts['face-detector'].get_first_face_bounding_box(ref_human_crop)
        edit_area_input_dict['edited_face_bbox'] = self.experts['face-detector'].get_first_face_bounding_box(edited_human_crop)
        self.logger.debug(f"Ref Face Bounding box: {edit_area_input_dict['ref_face_bbox']}")
        # get segmentation masks if needed
        for expert_name, expert in self.experts.items():
            region = self.experts_to_regions_map[expert_name]
            ref_mask = expert.get_mask(ref_human_crop)
            edited_mask = expert.get_mask(edited_human_crop)
            if ref_mask is None or edited_mask is None:
                self.logger.debug(f"🚨 [{region}] Segmentation mask is not exist!")
            edit_area_input_dict[f'ref_{region}_mask'] = ref_mask
            edit_area_input_dict[f'edited_{region}_mask'] = edited_mask

        return edit_area_input_dict

    def _prepare_metrics_input_dict(
        self,
        ref_image: Image.Image,
        edited_image: Image.Image,
        ref_human_crop: Image.Image,
        edited_human_crop: Image.Image,
        coords: List[Tuple[int, int, int, int]],
        scopes,
    ) -> Dict:
        metrics_input_dicts = {}
        for scope in scopes:
            if scope == 'edit_area':
                metrics_input_dicts['edit_area'] = self._prepare_edit_area_metrics_inputs(ref_human_crop, edited_human_crop)
            elif scope == 'unedit_area':
                metrics_input_dicts['unedit_area'] = {
                    "ref_image": ref_image,
                    "edited_image": edited_image,
                    "coords": coords
                }
            else:
                raise ValueError(f"Scope \"{scope}\" is not supported in HumanCenterPipeline.")

        return metrics_input_dicts
    
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
    
    def generate_rubric(self, edit_attributes):
        unedited_attributes = []
        for attr, attri_metrics in self.attributes_to_metrics_map.items():
            if attr in edit_attributes:
                continue
            for attri_metric in attri_metrics:
                if attri_metric in self.edit_area_metrics:
                    unedited_attributes.append(attri_metric)
        return unedited_attributes
    
    def __call__(self, input_dict, **kwargs):
        image_max_side = kwargs.get('image_max_side', 1024)
        valid_input_dict = self._prepare_input_dict(input_dict, image_max_side)

        # Step I: Parser & Grounder
        objects_dict, coords = self.parser_grounder(valid_input_dict, **kwargs)
        if objects_dict is None:
            return None, None
        objects_dict['edited_area_ratio'] = self.compute_edited_area_ratio(
            valid_input_dict['input_image'].size, coords, valid_input_dict['edit_task']
        )

        # Step II: Region-specific Metrics Computation
        # crop human region
        ref_human_crop, _ = self._crop_human_from_union_bbox(valid_input_dict['input_image'], coords)
        edited_human_crop, _ = self._crop_human_from_union_bbox(valid_input_dict['edited_image'], coords)
        # get rubrics
        measurement_rubric = self.generate_rubric(objects_dict['edit_attributes'])
        if objects_dict['edited_area_ratio'] < 0.9:
            measurement_rubric.extend(self.unedit_area_metrics)
        else:
            self.logger.debug(f"🚨 Skipping Unedit Area Measurement with edit area ratio is {objects_dict['edited_area_ratio']}")
        self.logger.debug(f'Generated Rubrics: {measurement_rubric}')
        # prepare metrics input dict
        metrics_input_dict = self._prepare_metrics_input_dict(
            ref_image=valid_input_dict['input_image'],
            edited_image=valid_input_dict['edited_image'],
            ref_human_crop=ref_human_crop,
            edited_human_crop=edited_human_crop,
            coords=coords,
            scopes=self.measure_scopes,
        )
        # compute metrics
        scores_dict = {}
        for metric_name, pipe_info in self.reject_sampling_pipes.items():
            scores_dict.setdefault(pipe_info['scope'], {})
            if metric_name not in measurement_rubric:
                scores_dict[pipe_info['scope']][metric_name] = None
                continue
            if (
                pipe_info['scope'] == 'edit_area'
                and (metrics_input_dict['edit_area']['ref_face_bbox'] is None or metrics_input_dict['edit_area']['edited_face_bbox'] is None)
                and (metric_name.split('_')[0] in ['body', 'face'])
            ):
                scores_dict[pipe_info['scope']][metric_name] = None
                continue
            score = pipe_info['pipe'](
                **metrics_input_dict[pipe_info['scope']],
                mask_mode=pipe_info['mask_mode'],
                metric=metric_name,
                **pipe_info['runtime_params'],
            )
            scores_dict[pipe_info['scope']][metric_name] = score
            self.logger.debug(f"Computed [{metric_name}] score: {score} for scope [{pipe_info['scope']}]")
        
        return scores_dict, objects_dict
                
                


