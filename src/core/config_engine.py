import os
import re
import yaml
from copy import deepcopy


class ConfigLoader(yaml.SafeLoader):
    pass

def _tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

ConfigLoader.add_constructor("!tuple", _tuple_constructor)


class ConfigEngine:

    VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, strict: bool = True):
        self.strict = strict

    def load(self, pipeline_path: str, user_path: str = None):

        user_config = self.load_yaml(user_path) if user_path else {}
        pipeline_cfg = self.load_yaml(pipeline_path)

        namespace = self.load_namespace(pipeline_cfg, pipeline_path)
        context = {
            "user_config": user_config,
            **user_config,
            **namespace,
            **pipeline_cfg
        }

        resolved_cfg = self.resolve_all(pipeline_cfg, context)
        resolved_cfg = self.normalize_default_init_config(resolved_cfg)

        return resolved_cfg

    def load_yaml(self, path: str):
        if path is None:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=ConfigLoader) or {}

    def load_namespace(self, cfg: dict, cfg_path: str):
        namespace = {}

        if "_base_" not in cfg:
            return namespace

        base = cfg["_base_"]
        cfg_dir = os.path.dirname(cfg_path)

        if not isinstance(base, dict):
            raise ValueError("_base_ must be a dict like {name: path}")

        for name, rel_path in base.items():
            full_path = os.path.join(cfg_dir, rel_path)
            namespace[name] = self.load_yaml(full_path)

        return namespace

    def resolve_all(self, cfg, context):
        prev = None
        cur = deepcopy(cfg)

        MAX_ITER = 10
        for _ in range(MAX_ITER):
            new = self._resolve_once(cur, context)
            if new == cur:
                return new
            cur = new

        raise ValueError("Possible circular reference detected in config")

    def _resolve_once(self, obj, context):
        if isinstance(obj, dict):
            return {k: self._resolve_once(v, context) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._resolve_once(v, context) for v in obj]

        elif isinstance(obj, str):
            return self._resolve_string(obj, context)

        return obj

    def _resolve_string(self, value: str, context):
        matches = self.VAR_PATTERN.findall(value)

        if not matches:
            return value

        if value.strip().startswith("${") and value.strip().endswith("}") and len(matches) == 1:
            return self._get_by_path(context, matches[0])

        for var in matches:
            replacement = self._get_by_path(context, var)
            value = value.replace(f"${{{var}}}", str(replacement))

        return value

    def _get_by_path(self, context, path: str):
        if path.startswith("env:"):
            return self._get_env_var(path[4:])

        keys = path.split(".")
        val = context

        for k in keys:
            if not isinstance(val, dict) or k not in val:
                if self.strict:
                    raise KeyError(f"Variable '{path}' not found in context")
                return None
            val = val[k]

        return val

    def _get_env_var(self, spec: str):
        if ":" in spec:
            env_name, default_value = spec.split(":", 1)
        else:
            env_name, default_value = spec, None

        env_value = os.environ.get(env_name)
        if env_value is not None and env_value != "":
            return yaml.safe_load(env_value)
        if default_value is not None:
            return yaml.safe_load(default_value)
        if self.strict:
            raise KeyError(f"Environment variable '{env_name}' is not set")
        return None

    def normalize_default_init_config(self, obj):
        if isinstance(obj, dict):
            normalized = {
                k: self.normalize_default_init_config(v) for k, v in obj.items()
            }

            if "default_config" in normalized and "init_config" in normalized:
                merged = self._merge_dict_config(
                    normalized.get("default_config"),
                    normalized.get("init_config")
                )
                normalized["init_config"] = merged
                normalized.pop("default_config", None)
            
            if "default_config" in normalized and "init_config" not in normalized:
                normalized["init_config"] = normalized.pop("default_config")

            return normalized

        if isinstance(obj, list):
            return [self.normalize_default_init_config(v) for v in obj]

        return obj

    def _merge_dict_config(self, default_cfg, init_cfg):
        base = deepcopy(default_cfg) if isinstance(default_cfg, dict) else {}
        override = init_cfg if isinstance(init_cfg, dict) else {}
        return self._deep_update(base, override)

    def _deep_update(self, base: dict, override: dict):
        for k, v in override.items():
            if (
                k in base
                and isinstance(base[k], dict)
                and isinstance(v, dict)
            ):
                base[k] = self._deep_update(base[k], v)
            else:
                base[k] = deepcopy(v)
        return base
    
if __name__ == "__main__":
    engine = ConfigEngine()
    config = engine.load(
        pipeline_path="configs/pipelines/human_centric/motion_change.yaml",
        user_path="configs/pipelines/user_config.yaml"
    )
    print(config)
