import os
import sys
import logging
import multiprocessing
import pandas as pd
try:
    from colorlog import ColoredFormatter
except ImportError:
    ColoredFormatter = None

LOGGER_NAME = 'auto_pipeline'
LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

class MainProcessOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _is_main_process()


def _is_main_process() -> bool:
    return multiprocessing.current_process().name == "MainProcess"


def _detach_handlers(logger: logging.Logger):
    # In forked workers, handlers can be inherited from parent process.
    # Remove them explicitly when non-main process logging is disabled.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def _set_main_process_filter(logger: logging.Logger, enabled: bool):
    existing_filters = [f for f in logger.filters if isinstance(f, MainProcessOnlyFilter)]
    if enabled and not existing_filters:
        logger.addFilter(MainProcessOnlyFilter())
    elif not enabled and existing_filters:
        for f in existing_filters:
            logger.removeFilter(f)


def basic_logger_init(log_dir: str, level: str, main_process_only: bool = False):
    if main_process_only and not _is_main_process():
        return

    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir,
        f'run_{pd.Timestamp.now().strftime("%m%d_%H%M%S")}.log'
    )

    logging.basicConfig(
        filename=log_filename,
        level=LEVEL_MAP.get(level, logging.DEBUG),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logging.info("日志系统初始化完成。")

def logger_init(log_dir: str, level: str, main_process_only: bool = False):
    """
    初始化一个专用的、对分布式环境友好的“命名记录器”。
    
    Args:
        log_dir (str): 日志文件存放目录。
        level (str): 日志级别 (例如 'info', 'debug')。
        main_process_only (bool): 为 True 时，仅主进程输出日志。
        app_name (str): 你项目的专属 logger 名称 (例如 "eval_bench")。
    """

    # 1) 获取 local_rank，默认为 "0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_proc = _is_main_process()

    # 2) 获取命名 logger
    logger = logging.getLogger(LOGGER_NAME)

    # 3) 设置 logger 级别
    log_level = LEVEL_MAP.get(level, logging.DEBUG)
    logger.setLevel(log_level)

    # 4) 阻止日志传播到 root logger，避免记录其他库日志
    logger.propagate = False
    _set_main_process_filter(logger, main_process_only)

    # 5) 可选：仅主进程打印日志
    if main_process_only and not is_main_proc:
        _detach_handlers(logger)
        if not logger.hasHandlers():
            logger.addHandler(logging.NullHandler())
        return logger

    # 6) 在 rank0 上设置 handlers（并防重复）
    if local_rank == 0 and not logger.hasHandlers():
        
        # --- 你原有的逻辑 ---
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(
            log_dir,
            f'run_{pd.Timestamp.now().strftime("%m%d_%H%M%S")}.log'
        )
        # --- 结束原有逻辑 ---
        
        # 7) 定义日志格式
        if ColoredFormatter is not None:
            formatter = ColoredFormatter(
                "%(log_color)s[%(asctime)s] - %(levelname)-8s%(reset)s %(message_log_color)s%(message)s",
                datefmt=None,
                reset=True,
                log_colors={
                    'DEBUG':    'bold_light_blue',
                    'INFO':     'bold_light_cyan',
                    'WARNING':  'bold_light_yellow',
                    'ERROR':    'bold_light_red',
                    'CRITICAL': 'bold_light_red,bg_white',
                },
                secondary_log_colors={
                    'message': {
                        'DEBUG':    'white',
                        'INFO':     'white',
                        'WARNING':  'white',
                        'ERROR':    'white',
                        'CRITICAL': 'white',
                    }
                },
                style='%'
            )
        else:
            formatter = logging.Formatter(
                "[%(asctime)s] - %(levelname)-8s %(message)s"
            )
        
        # 8) 创建 FileHandler (写入文件)
        fh = logging.FileHandler(log_filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # 9) 同时创建 StreamHandler (打印到控制台)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(log_level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        logger.info(f"日志系统初始化完成 (Rank {local_rank})。日志将保存到 {log_filename}")

    elif local_rank != 0 and not logger.hasHandlers():
        # 对于非 rank0 进程，添加 NullHandler 抑制“无 handler”警告
        logger.addHandler(logging.NullHandler())
    
    # 10) 返回配置好的 logger 实例
    return logger

def get_logger():
    return logging.getLogger(LOGGER_NAME)
