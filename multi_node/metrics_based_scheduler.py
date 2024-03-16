from dataclasses import dataclass
from typing import List
import random
from data_parallel_request_cache import CustomRuntimeSelector
from model_runtime_manager import EndpointRuntimeInterface
import concurrent.futures
import time

@dataclass
class MetricData:
    waiting_queue_len: int
    running_req_len: int
    prefix_match_len: int
    token_kv_available_size: int
    evicatable_size: int
    tree_cache_metrics_hit: int
    tree_cache_metrics_total: int
    input_len: int

    @staticmethod
    def from_dict(input_dict):
        return MetricData(
            waiting_queue_len=input_dict["waiting_queue_len"],
            running_req_len=input_dict["running_req_len"],
            prefix_match_len=input_dict["prefix_match_len"],
            token_kv_available_size=input_dict["token_kv_available_size"],
            evicatable_size=input_dict["evicatable_size"],
            tree_cache_metrics_hit=input_dict["tree_cache_metrics_hit"],
            tree_cache_metrics_total=input_dict["tree_cache_metrics_total"],
            input_len=input_dict["input_len"]
        )

@dataclass
class LongestPrefixMatchSelector(CustomRuntimeSelector):
    runtimes: List[EndpointRuntimeInterface]
    def __post_init__(self):
        self.metrics_dict = []

    def runtime_selector(self, text: str):
        # Send a request to each runtime
        start_time = time.time()
        metrics = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(runtime.metrics_request, text) for runtime in self.runtimes]
            metrics = [MetricData.from_dict(future.result()) for future in concurrent.futures.as_completed(futures)]

        # Handle the case where the prefix match len/input length is really small less than 1%. Pick randomly
        if all(metric.prefix_match_len / metric.input_len < 0.02 for metric in metrics):
            selected_runtime =  random.randint(0, len(self.runtimes) - 1) # Randomly select
        else:
            # Pick the runtime with the highest prefix match len/input length
            max_prefix_match = max(range(len(self.runtimes)), key=lambda i: metrics[i].prefix_match_len / metrics[i].input_len)

            selected_runtime = max_prefix_match
        
        self.metrics_dict.append({
            "text": text[:100],
            "metrics": [metric.__dict__ for metric in metrics],
            "selected_runtime": selected_runtime,
            "overhead": time.time() - start_time
        })
            
        return selected_runtime
