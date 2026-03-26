import time
import requests
from openai import OpenAI
from . import CLIENT_REGISTRY
from .base_client import BaseClient
from common_utils.logging_util import get_logger
logger = get_logger()

@CLIENT_REGISTRY.register("vllm")
class vLLMOnlineClient(BaseClient):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        assert kwargs.get('ip_address'), "ip_address is required for vLLMOnlineClient"
        assert kwargs.get('port'), "port is required for vLLMOnlineClient"
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{kwargs.get('ip_address')}:{kwargs.get('port')}/v1",
            timeout=kwargs.get('timeout', 600),
        )
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.retries = kwargs.get('retries', 3)
        self.temperature = kwargs.get('temperature', 0.7)
        self.extra_body = kwargs.get('extra_body', {})

    def call_model(self, messages):
        try_count = 0
        while True:
            try:
                if "Qwen3-8B" in self.model_name:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        seed=42+try_count,
                        extra_body=self.extra_body,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        seed=42+try_count,
                    )
                return response.choices[0].message.content
            except requests.exceptions.RequestException as e:
                try_count += 1
                logger.info(f"🔁 Request failed: {e}. Retrying...")
                time.sleep(2)
                if try_count > self.retries:
                    return None

# class VLMClient(VLMPrompter, BasevLLMOnlineClient):
#     def __init__(self, ip_address, port, served_model_name, **kwargs):
#         super().__init__(ip_address, port, served_model_name, **kwargs)
#         self.task = kwargs.get('task', 'chat')
#         self.assessment_type = None
#         self.prompt_version = None

#         if self.task in ['pair-judge', 'viescore']:
#             self.assessment_type = kwargs.get('assessment_type', 'visual_consistency')
#             self.prompt_version = kwargs.get('prompt_version', 'v1')

#     def __call__(self, image_dict, instruction, **kwargs):
#         messages = self.prepare_messages(
#             image_dict,
#             instruction,
#             task=self.task,
#             assessment_type=self.assessment_type,
#             prompt_version=self.prompt_version,
#             **kwargs
#         )

#         return self.call_model(messages)

# class LLMClient(LLMPrompter, BasevLLMOnlineClient):
#     def __init__(self, ip_address, port, served_model_name, **kwargs):
#         super().__init__(ip_address, port, served_model_name, **kwargs)

#     def __call__(self, instruction, edit_task_type, **kwargs):
#         messages = self.prepare_messages(
#             instruction,
#             edit_task_type,
#             **kwargs
#         )

#         return self.call_model(messages)

