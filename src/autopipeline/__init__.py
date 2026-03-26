try:
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
except ImportError:
    initialize_config_module = None
    GlobalHydra = None

if initialize_config_module is not None and GlobalHydra is not None:
    if not GlobalHydra.instance().is_initialized():
        initialize_config_module("autopipeline/components/primitives/sam2", version_base="1.2")