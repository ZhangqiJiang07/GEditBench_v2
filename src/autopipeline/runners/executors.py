from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed
import threading
from common_utils.logging_util import get_logger

logger = get_logger()


_PROCESS_RUNTIME = None


def _init_process_runtime(pipeline_loader):
    global _PROCESS_RUNTIME
    _PROCESS_RUNTIME = Runtime(pipeline_loader)
    _PROCESS_RUNTIME.init_process()


def _run_in_process(worker, item_key, item_input, **kwargs):
    if _PROCESS_RUNTIME is None:
        raise RuntimeError("Process runtime is not initialized.")
    return worker(item_key, item_input, _PROCESS_RUNTIME, **kwargs)


class Runtime:
    def __init__(self, pipeline_loader):
        self.pipeline_loader = pipeline_loader

        self._process_pipeline = None
        self._thread_local = threading.local()

    def init_process(self):
        self._process_pipeline = self.pipeline_loader.load()
    
    def get_pipeline(self):
        if self._process_pipeline is not None:
            return self._process_pipeline
        if not hasattr(self._thread_local, "pipeline"):
            self._thread_local.pipeline = self.pipeline_loader.load()
        return self._thread_local.pipeline

class BaseExecutor:
    def run(self, items, dataset, worker, pipeline_loader):
        raise NotImplementedError

class ThreadExecutor(BaseExecutor):
    def __init__(self, max_workers=50):
        self.max_workers = max_workers

    def run(self, items, dataset, worker, pipeline_loader):
        runtime = Runtime(pipeline_loader)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item_key = {
                executor.submit(
                    worker,
                    item_key,
                    dataset.get_item(item_key),
                    runtime,
                    benchmark_name=getattr(dataset, "benchmark_name", None),
                ): item_key
                for item_key in items
            }

            for future in tqdm(as_completed(future_to_item_key), total=len(future_to_item_key), desc="Processing results"):
                item_key = future_to_item_key[future]
                try:
                    yield future.result()
                except Exception as e:
                    logger.error(f"Thread worker failed for item {item_key}: {e}")
                    continue

class ProcessExecutor(BaseExecutor):
    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def run(self, items, dataset, worker, pipeline_loader):
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_process_runtime,
            initargs=(pipeline_loader,),
        ) as executor:

            future_to_item_key = {
                executor.submit(
                    _run_in_process,
                    worker,
                    item_key,
                    dataset.get_item(item_key),
                    benchmark_name=getattr(dataset, "benchmark_name", None),
                ): item_key
                for item_key in items
            }

            for future in tqdm(as_completed(future_to_item_key), total=len(future_to_item_key), desc="Processing results"):
                item_key = future_to_item_key[future]
                try:
                    yield future.result()
                except Exception as e:
                    logger.error(f"🚨 Process worker failed for item {item_key}: {e}")
                    continue
