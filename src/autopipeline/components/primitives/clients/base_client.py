import re
import json
import json_repair
from abc import ABC, abstractmethod
from typing import Any, Dict
from common_utils.logging_util import get_logger
from common_utils.json_util import extract_json_block
logger = get_logger()

class BaseClient(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def call_model(self, messages) -> str:
        pass

    def parse_response_to_json(self, response: str) -> Any:
        json_str = extract_json_block(response)

        if not json_str:
            # fallback: try whole text repair
            try:
                return json_repair.loads(response)
            except:
                return None

        try:
            return json_repair.loads(json_str)
        except Exception as e:
            logger.error(f"Parse JSON error: {e}")
            return None