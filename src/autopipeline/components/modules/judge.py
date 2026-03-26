import numpy as np
from typing import Dict, Any

from . import BasePipe, PIPE_REGISTRY
from ..primitives.clients import CLIENT_REGISTRY
from ..primitives import PROMPT_ADAPTER_REGISTRY
from schemas.prompt_template import PromptTemplate
import schemas.pipeline_io as pipeline_io_schemas
from common_utils.logging_util import get_logger
logger = get_logger()


def build_client(
    backend: str,
    model_name: str = None,
    ip_address: str = None,
    port: int = None,
    base_url: str = None,
    api_key: str = None,
    # max_tokens: int = 2048,
    # retries: int = 3,
    # timeout: int = 600,
    # temperature: float = 0.7,
    # extra_body: Dict[str, Any] = None,
    **kwargs
):
    client_class = CLIENT_REGISTRY.get(backend)
    if not client_class:
        raise ValueError(f"Backend '{backend}' is not supported")
    return client_class(
        model_name=model_name,
        ip_address=ip_address,
        port=port,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )

class ClientPipe(BasePipe):
    def __init__(self, prompt_template: PromptTemplate, client_cfg: Dict[str, Any]):
        self.prompt_template = prompt_template
        self.client = build_client(**client_cfg)
        self.prompt_style = "google_style" if client_cfg.get("backend") == "google" else "openai_style"
        prompt_adapter_cls = PROMPT_ADAPTER_REGISTRY.get(self.prompt_style)
        self.prompt_adapter = prompt_adapter_cls()
        if not self.prompt_adapter:
            raise ValueError(f"Prompt style '{self.prompt_style}' is not supported")
        self.output_schema = self._get_schema(self.prompt_template.metadata.get('output_schema_name', None))
        self.input_schema = self._get_schema(self.prompt_template.metadata.get('input_schema_name', None))

    def _get_schema(self, schema_name: str = None):
        if not schema_name:
            return None
        schema_class = getattr(pipeline_io_schemas, schema_name, None)
        return schema_class

    def _get_cleaned_input_by_schema(self, raw_input: Dict[str, Any]):
        if self.input_schema:
            try:
                cleaned_input = self.input_schema(**raw_input).dict()
                return cleaned_input
            except Exception as e:
                logger.warning(
                    f"Schema validation errors: {e}. Proceeding with raw input without schema validation."
                )
                return raw_input
        else:
            return raw_input
    
    def __call__(self, messages):
        raise NotImplementedError("Subclasses of ClientPipe must implement the __call__ method.")


@PIPE_REGISTRY.register("pairwise-judge")
class PairwiseJudge(ClientPipe):
    def __init__(self, prompt_template: PromptTemplate, client_cfg: Dict[str, Any]):
        super().__init__(prompt_template, client_cfg)

    def _is_valid_winner(self, winner_str):
        if winner_str.lower() in ['image_a', 'image a', 'a']:
            return "Image A"
        elif winner_str.lower() in ['image_b', 'image b', 'b']:
            return "Image B"
        elif winner_str.lower() in ['tie', 'equal', 'both', 'none']:
            return "Tie"
        return False
    
    def __call__(self, input_dict: Dict[str, Any], **kwargs):
        res = {
            'type': 'pairwise_comparison',
            'value': "Failed",
        }
        cleaned_input_dict = self._get_cleaned_input_by_schema(input_dict)
        messages = self.prompt_adapter.build_payload(self.prompt_template, **cleaned_input_dict)
        retry_count = 0
        while True:
            logger.debug(f"📡 Calling model for pairwise comparison... (Attempt {retry_count + 1})")
            response = self.client.call_model(messages)
            json_response = self.client.parse_response_to_json(response)
            logger.debug(f"[RESPONSE]: {json_response}")

            # check response format according to self.prompt_template.metadata['output_schema_name']
            if self.output_schema:
                try:
                    self.output_schema(**json_response)
                    winner = self._is_valid_winner(json_response.get("winner", ""))
                    if winner:
                        res['value'] = winner
                        res['meta'] = {"raw_response": response}
                        return res
                    else:
                        logger.info(f"Winner field in response is not valid. Retrying...")
                except Exception as e:
                    logger.info(f"Response parsing failed: {e}. Retrying...")

            else:
                if json_response is not None and self._is_valid_winner(json_response.get("winner", "")):
                    res['value'] = self._is_valid_winner(json_response.get("winner", ""))
                    res['meta'] = {"raw_response": response}
                    return res
                else:
                    logger.info(f"Response is not valid JSON. Retrying...")

            retry_count += 1
            if retry_count > self.client.retries:
                logger.info(f"Exceeded maximum retries for valid JSON response. Returning None.")
                res['meta'] = {"raw_response": response}
                return res


@PIPE_REGISTRY.register("viescore")
class VIEscorePipe(ClientPipe):
    def __init__(self, prompt_template: PromptTemplate, client_cfg: Dict[str, Any]):
        super().__init__(prompt_template, client_cfg)

    def _parse_viescore(self, json_response: Dict[str, Any]) -> float:
        score = json_response.get("score", None)
        try:
            if isinstance(score, list):
                if self.prompt_template.version == "v2": # for Editscore prompt
                    if "instruction_following" in self.prompt_template.prompt_id:
                        return float(score[0])
                    elif "visual_quality" in self.prompt_template.prompt_id:
                        return float(min(score))
                    else:
                        return float(score[1])
                else:
                    raise ValueError(
                        f"Unexpected list format for score in VIEscorePipe with prompt if {self.prompt_template.prompt_id} version: {self.prompt_template.version}"
                    )
            else: # UnicEdit prompt
                return float(score)
        except Exception as e:
            logger.info(f"Error parsing VIEscore from response: {e}")
            return None

    def score_single_input(self, messages):
        retry_count = 0
        while True:
            response = self.client.call_model(messages)
            json_response = self.client.parse_response_to_json(response)

            if self.output_schema:
                try:
                    self.output_schema(**json_response)
                    vie_score = self._parse_viescore(json_response)
                    if vie_score is not None:
                        return vie_score, response
                    else:
                        logger.info(f"VIEscore field in response is not valid. Retrying...")
                except Exception as e:
                    logger.info(f"Response parsing failed: {e}. Retrying...")

            else:
                if json_response is not None:
                    vie_score = self._parse_viescore(json_response)
                    if vie_score is not None:
                        return vie_score, response
                    else:
                        logger.info(f"VIEscore field in response is not valid. Retrying...")
                else:
                    logger.info(f"Response is not valid JSON. Retrying...")

            retry_count += 1
            if retry_count > self.client.retries:
                logger.info(f"Exceeded maximum retries for valid JSON response. Returning None.")
                return None, response

    def __call__(self, input_dict: Dict[str, Any], **kwargs):
        edited_images_num = len(input_dict.get("edited_images", []))
        assert (edited_images_num >= 1 and edited_images_num <= 2), f"Unsupported number of edited images: {edited_images_num}. Only 1 or 2 edited images are supported."
        vie_scores = [None] * edited_images_num
        raw_responses = {f"edited_image_{i}": None for i in range(edited_images_num)}
        res = {}
        if edited_images_num == 1:
            res['type'] = 'single_score'
        else:
            res['type'] = 'pairwise_comparison'
        res['value'] = "Failed"

        for i, edit_image in enumerate(input_dict['edited_images']):
            temp_input_dict = {
                "instruction": input_dict["instruction"],
                "input_image": input_dict["input_image"],
                "edited_image": edit_image,
            }
            messages = self.prompt_adapter.build_payload(self.prompt_template, **temp_input_dict)
            _vie_score, _response = self.score_single_input(messages)
            vie_scores[i] = _vie_score
            raw_responses[f"edited_image_{i}"] = _response

        res['meta'] = {
            "vie_score": vie_scores,
            "raw_response": raw_responses
        }
        if any(score is None for score in vie_scores):
            return res
        
        if edited_images_num == 1:
            res['value'] = vie_scores[0]
            return res

        if vie_scores[0] > vie_scores[1]:
            winner = "Image A"
        elif vie_scores[0] < vie_scores[1]:
            winner = "Image B"
        else:
            winner = "Tie"
        res['value'] = winner

        return res
