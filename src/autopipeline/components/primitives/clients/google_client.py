import random
import time
from . import CLIENT_REGISTRY
from .base_client import BaseClient
from google import genai
from google.genai import types
from common_utils.logging_util import get_logger
logger = get_logger()

@CLIENT_REGISTRY.register("google")
class GoogleAPIClient(BaseClient):
    def __init__(self, model_name = 'gemini-3-pro-native', **kwargs):
        super().__init__(model_name, **kwargs)
        assert kwargs.get('base_url'), "base_url is required for GoogleAPIClient"
        assert kwargs.get('api_key'), "api_key is required for GoogleAPIClient"
        self.client = genai.Client(
            http_options={
                'base_url': kwargs.get('base_url'),
                'api_version': kwargs.get('api_version', 'v1alpha'),
            },
            api_key=kwargs.get('api_key')
        )
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.retries = kwargs.get('retries', 3)

    def call_model(self, messages):
        # from google.genai import types
        base_delay = 2
        for attempt in range(self.retries):
            try:
                # print(f"[DEBUG] GoogleAPIClient sending messages: {messages}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.max_tokens
                    )
                )
                
                if response and response.text:
                    return response.text
                else:
                    logger.warning(f"Empty response received on attempt {attempt + 1}")

            except Exception as e:
                error_msg = str(e)
                
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)                
                if "429" in error_msg or "Resource has been exhausted" in error_msg:
                    logger.warning(f"⚠️ Rate Limit hit (429). Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{self.retries})")
                else:
                    logger.error(f"❌ API Error: {error_msg}. Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{self.retries})")
                
                time.sleep(sleep_time)

        logger.error(f"Failed to get response after {self.retries} attempts.")
        return None