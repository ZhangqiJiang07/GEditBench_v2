import os
import logging
import pandas as pd
from autotrain.constants import LOG_DIR

def logger_init():
    local_rank = int(os.environ.get("LOCAL_RANK", None))
    log_path = os.path.join(LOG_DIR, pd.Timestamp.now().strftime('%m%d_%H%M%S'))

    os.makedirs(log_path, exist_ok=True)
    log_filename = os.path.join(
        log_path,
        f'rank_{local_rank}.log'
    )

    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logging.debug("日志系统初始化完成。")
