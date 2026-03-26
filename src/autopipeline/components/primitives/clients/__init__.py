from core.registry import Registry


CLIENT_REGISTRY = Registry(name="Clients", enable_regex=False)

from . import google_client
from . import openai_client
from . import vllm_client