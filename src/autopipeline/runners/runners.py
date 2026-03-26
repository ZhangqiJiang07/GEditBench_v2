from core.cache_manager import generate_cache_key
from common_utils.logging_util import get_logger
logger = get_logger()


class Runner:
    def __init__(
        self,
        pipeline_loader,
        worker,
        executor,
        cache_manager,
        dataset
    ):
        self.pipeline_loader = pipeline_loader
        self.worker = worker
        self.executor = executor
        self.cache_manager = cache_manager
        self.dataset = dataset

    def _is_valid_result(self, result):
        if self.worker.pipeline_name == "vlm-as-a-judge" and result.get("winner") == "Failed":
            return False
        return True
    
    def run(self):
        item_to_process, all_results = self.dataset.load_cache(self.cache_manager)

        if not item_to_process:
            logger.info("No items to process. All results loaded from cache.")
            return all_results
        
        # results = self.executor.run(
        #     items=item_to_process,
        #     dataset=self.dataset,
        #     worker=self.worker,
        #     pipeline_loader=self.pipeline_loader
        # )

        for item_key, res_dict in self.executor.run(
            items=item_to_process,
            dataset=self.dataset,
            worker=self.worker,
            pipeline_loader=self.pipeline_loader
        ):
            try:
                if self._is_valid_result(res_dict):
                    all_results[item_key] = res_dict
                    # logger.debug(f"🪄 NM Cached!")
                    self.cache_manager.append(generate_cache_key(item_key), res_dict)
                    logger.debug(f"📦 Successfully Cached!")
            except Exception as e:
                logger.error(f"Error processing result for item {item_key}: {e}")
        return all_results
        
