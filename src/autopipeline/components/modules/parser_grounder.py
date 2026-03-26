
from typing import Dict, Any

from ..primitives import PROMPT_ADAPTER_REGISTRY
from . import BasePipe, PIPE_REGISTRY
from .judge import build_client
from prompts.prompt_manager import PromptTemplate
import schemas.pipeline_io as pipeline_io_schemas
from common_utils.json_util import parse_Qwen3_VL_coordinates
from common_utils.logging_util import get_logger
logger = get_logger()


@PIPE_REGISTRY.register("parser-grounder")
class ParserGrounderPipe(BasePipe):
    def __init__(self, config, prompt_template_dict: Dict[str, PromptTemplate]):
        self.config = config
        backend = dict(config['instruction_parser']['init_config'])['backend']
        self.prompt_template_parser = prompt_template_dict.get("instruction_parser")
        self.instruction_parser = build_client(**dict(config['instruction_parser']['init_config']))
        self.prompt_template_grounder = prompt_template_dict.get("general_grounder")
        self.object_grounder = build_client(**dict(config['general_grounder']['init_config']))
        self.prompt_style = "google_style" if backend == "google" else "openai_style"
        prompt_adapter_cls = PROMPT_ADAPTER_REGISTRY.get(self.prompt_style)
        self.prompt_adapter = prompt_adapter_cls()
        if not self.prompt_adapter:
            raise ValueError(f"Prompt style '{self.prompt_style}' is not supported")
        
        self.instruction_parser_output_schema = self._get_schema(self.prompt_template_parser.metadata.get('output_schema_name', None))
        self.object_grounder_output_schema = self._get_schema(self.prompt_template_grounder.metadata.get('output_schema_name', None))
        logger.info(f"🚀 Initialized ParserGrounderPipe with instruction parser backend [{backend}] and prompt style [{self.prompt_style}].")


    def _get_schema(self, schema_name: str = None):
        if not schema_name:
            return None
        schema_class = getattr(pipeline_io_schemas, schema_name, None)
        return schema_class
    
    def _check_format_by_schema(self, raw_data: Dict[str, Any], schema_class):
        try:
            schema_class(**raw_data)
            return True
        except Exception as e:
            logger.error(f"Output does not conform to schema {schema_class.__name__}: {e}")
        return False

    def _prepare_grounding_inputs(self, input_dict, objects_dict, edit_task_type):
        # object center
        if edit_task_type in ['SUBJECT_ADD']:
            images_to_process = [input_dict['edited_image']]
            object_lists_for_grounding = [objects_dict['edited_objects']]
        elif edit_task_type in ['SUBJECT_REMOVE', 'COLOR_ALTER', 'MATERIAL_ALTER']:
            images_to_process = [input_dict['input_image']]
            object_lists_for_grounding = [objects_dict['edited_objects']]
        elif edit_task_type in ['SUBJECT_REPLACE']:
            images_to_process = [input_dict['input_image'], input_dict['edited_image']]
            object_lists_for_grounding = [objects_dict['edited_objects'], objects_dict['generated_objects']]
        elif edit_task_type in ['OBJECT_EXTRACTION', 'OREF', 'SIZE_ADJUSTMENT']:
            images_to_process = [input_dict['input_image'], input_dict['edited_image']]
            object_lists_for_grounding = [objects_dict['edited_objects'], objects_dict['edited_objects']]

        # human center
        elif edit_task_type in ['PS_HUMAN', 'MOTION_CHANGE']:
            images_to_process = [input_dict['input_image'], input_dict['edited_image']]
            object_lists_for_grounding = [[objects_dict['edited_subjects']], [objects_dict['edited_subjects']]]
        elif edit_task_type in ['CREF']:
            images_to_process = [input_dict['input_image']]
            object_lists_for_grounding = [objects_dict['edited_subjects']]
        else:
            raise ValueError(f"Unsupported edit_task_type: {edit_task_type}")
        return images_to_process, object_lists_for_grounding

    def _valied_grounding_output(self, grounding_output, object_list):
        if grounding_output is None or not isinstance(grounding_output, list):
            logger.error(f"👻 Grounding output is invalid: {grounding_output}")
            return False
        for item in grounding_output:
            if item.get('label', 'No labels!') not in object_list:
                logger.error(f"🎼 Grounding output contains label not in object list. Label: {item.get('label', 'No labels!')}, Object List: {object_list}")
                return False
        return True
    
    def __call__(self, input_dict: Dict[str, Any], **kwargs):
        # print(f"[DEBUG] Received input_dict: {input_dict.keys()}")
        instruction_parsing_messages = self.prompt_adapter.build_payload(
            self.prompt_template_parser, **input_dict
        )
        # print(f"Instruction parsing messages: {instruction_parsing_messages}")
        for attempt in range(self.instruction_parser.retries):
            # Instruction Parsing Phase
            extraction_text = self.instruction_parser.call_model(instruction_parsing_messages)
            # logger.debug(f"INSTRUCTION: {input_dict.get('instruction', '')}")
            # logger.debug(f"Instruction parsing text: {extraction_text}")
            extraction_json_response = self.instruction_parser.parse_response_to_json(extraction_text)
            if not self._check_format_by_schema(extraction_json_response, self.instruction_parser_output_schema):
                logger.debug(f"Instruction parsing output does not conform to schema. Retrying... (Attempt {attempt + 1})")
                if attempt == self.instruction_parser.retries - 1:
                    logger.debug(f"Max retries reached for instruction parsing. Returning None.")
                    return None, None
                continue

            # Grounding Phase
            images_to_process, object_lists_for_grounding = self._prepare_grounding_inputs(
                input_dict, extraction_json_response, input_dict.get('edit_task', '')
            )
            img_obj_bbox_list = [[] for _ in object_lists_for_grounding]
            for idx, (img, obj_list) in enumerate(zip(images_to_process, object_lists_for_grounding)):
                object_grounding_messages = self.prompt_adapter.build_payload(
                    self.prompt_template_grounder, **{
                        "instruction": str(obj_list),
                        "grounding_image": img
                    }
                )
                # print(self.prompt_template_grounder.user_prompt)
                # logger.debug(f"🚀 [DEBUG] Object grounding messages for image {idx}: {object_grounding_messages}")
                grounding_text = self.object_grounder.call_model(object_grounding_messages)
                grounding_json_text = self.object_grounder.parse_response_to_json(grounding_text)
                if not self._valied_grounding_output(grounding_json_text, obj_list):
                    break

                logger.debug(f"Object grounding text: {grounding_text}")
                coords = parse_Qwen3_VL_coordinates(
                    grounding_text, img.size[1], img.size[0]
                )
                if coords is not None:
                    img_obj_bbox_list[idx] = coords

            if any([len(bboxes) == 0 for bboxes in img_obj_bbox_list]):
                logger.debug(f"Grounding failed to extract valid bounding boxes. Retrying... (Attempt {attempt + 1})")
                if attempt == self.instruction_parser.retries - 1:
                    logger.debug(f"Max retries reached for grounding. Returning None.")
                    return None, None
                continue

            extraction_json_response['bboxes'] = img_obj_bbox_list
            logger.debug(f"Extracted objects and bounding boxes: {extraction_json_response}")
            break

        all_coords = []
        for img_coords in img_obj_bbox_list:
            all_coords.extend(img_coords)

        return extraction_json_response, all_coords






