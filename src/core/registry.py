import re
from typing import Callable, Dict, Any, Optional
from common_utils.logging_util import get_logger

logger = get_logger()

class Registry:
    """
    通用的组件注册表。
    支持精确匹配 (exact match) 和 正则匹配 (regex match)。
    """
    def __init__(self, name: str, enable_regex: bool = False):
        self.name = name
        self.enable_regex = enable_regex
        self._module_dict: Dict[str, Any] = {}

    def register(self, name_or_pattern: Optional[str] = None) -> Callable:
        """
        通用的注册装饰器。如果不传名字，默认使用类名或函数名。
        """
        def decorator(obj: Any):
            # 获取注册的 Key
            key = name_or_pattern if name_or_pattern else obj.__name__
            
            if key in self._module_dict:
                logger.warning(f"[{self.name}] Overwriting existing key: {key}")
                
            self._module_dict[key] = obj
            return obj
            
        return decorator

    def get(self, query_name: str) -> Any:
        """
        获取注册的组件。根据 enable_regex 决定匹配策略。
        """
        # 1. 如果开启了正则匹配（适用于模型 Adapter）
        if self.enable_regex:
            for pattern, obj in self._module_dict.items():
                if re.match(pattern, query_name, re.IGNORECASE):
                    return obj
        # 2. 精确匹配（适用于 Dataset, Metrics 等）
        else:
            if query_name in self._module_dict:
                return self._module_dict[query_name]

        # 没找到时抛出清晰的异常
        raise KeyError(
            f"[{self.name}] '{query_name}' not found. "
            f"Available: {list(self._module_dict.keys())}"
        )

    def registered_keys(self):
        return list(self._module_dict.keys())