import os
import json
import threading
import logging
import hashlib
import numpy as np 
from typing import Any, Dict, Optional
from common_utils.logging_util import get_logger
logger = get_logger()

def generate_cache_key(pair_key):
    return hashlib.sha256(pair_key.encode("utf-8")).hexdigest()


class NumpyEncoder(json.JSONEncoder):
    """ 自定义编码器，用于处理Numpy数据类型 """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class CacheManager:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.lock = threading.Lock()
        self.cache = self._load()

    def _load(self) -> Dict[str, Any]:
        cache = {}
        if not os.path.exists(self.cache_file):
            logger.info(
                f"Cache file not found at {self.cache_file}. A new one will be created."
            )
            return cache
        
        with open(self.cache_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    cache[data["key"]] = data["result"]
                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping corrupted line {i + 1} in cache file: {line.strip()}"
                    )
        logger.info(f"Loaded {len(cache)} items from {self.cache_file}.")
        return cache
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def append(self, key: str, result: Any):
        with self.lock:
            self.cache[key] = result
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"key": key, "result": result}, ensure_ascii=False)
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())