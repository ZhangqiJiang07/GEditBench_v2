import os
# import sys
# sys.path.append('data/src')
import yaml
from pathlib import Path
from typing import Dict
from schemas.prompt_template import PromptTemplate
from common_utils.logging_util import get_logger
logger = get_logger()


class PromptAssetStore:
    """Load and cache prompt YAML assets via prompt_id + version."""

    def __init__(self, assets_dir: str | Path = 'assets'):
        self.base_dir = Path(__file__).parent / assets_dir
        self._cache: Dict[str, PromptTemplate] = {}
    
    def _get_file_path(self, prompt_id: str, version: str) -> Path:
        filename = os.path.join(prompt_id, f"{version}.yaml")
        return self.base_dir / filename
    
    def get_prompt(self, prompt_id: str, version: str) -> PromptTemplate:
        prompt_key = prompt_id.replace('/', '_')
        cache_key = f"{prompt_key}_{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self._get_file_path(prompt_id, version)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {file_path}. "
                f"Please ensure '{prompt_id}' version '{version}' exists."
            )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {file_path}: {e}")
            raise ValueError(f"Invalid YAML format in {file_path}") from e

        try:
            template = PromptTemplate(
                prompt_id=raw_data.get("prompt_id", prompt_id),
                version=raw_data.get("version", version),
                system_prompt=raw_data.get("system_prompt", ""),
                user_prompt=raw_data["user_prompt"],
                metadata=raw_data.get("metadata", {})
            )
        except KeyError as e:
            raise KeyError(f"Missing required field {e} in {file_path}")

        self._cache[cache_key] = template
        return template

    def clear_cache(self):
        self._cache.clear()



if __name__ == "__main__":
    # Example usage
    store = PromptAssetStore(assets_dir="./assets")
    prompt_template = store.get_prompt(prompt_id="vlm/assessment/visual_quality/pairwise", version="v1")
    print(prompt_template.system_prompt)
    print(prompt_template.user_prompt)