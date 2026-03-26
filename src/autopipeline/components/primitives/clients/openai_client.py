import time
import requests
from . import CLIENT_REGISTRY
from .base_client import BaseClient
from common_utils.logging_util import get_logger
logger = get_logger()

@CLIENT_REGISTRY.register("api")
class OpenAIAPIClient(BaseClient):
    def __init__(self,
        # base_url: str = "https://api.openai.com/v1/chat/completions",
        # api_key: str = "YOUR_OPENAI_API_KEY",
        model_name: str = "gpt-4o",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        assert kwargs.get('base_url'), "base_url is required for OpenAIAPIClient"
        assert kwargs.get('api_key'), "api_key is required for OpenAIAPIClient"
        self.base_url = kwargs.get('base_url')
        self.api_key = kwargs.get('api_key')
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.retries = kwargs.get('retries', 3)
        self.timeout = kwargs.get('timeout', 600)

    def call_model(self, messages) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        print(f"📡 Calling OpenAI API at {self.base_url} with model '{self.model_name}'...")
        
        body = {"model": self.model_name, "stream": False, "max_tokens": self.max_tokens, "messages": messages}
        try_count = 0
        while True:
            try:
                logger.debug(f"🔁 API call attempt {try_count + 1}")
                resp = requests.post(self.base_url, headers=headers, json=body, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                try_count += 1
                logger.debug(f"🔁 API call retry, error: {e}")
                time.sleep(2)
                if try_count > self.retries:
                    return None
