from dataclasses import dataclass
from typing import List
import random
from data_parallel_request_cache import CustomRuntimeSelector
from model_runtime_manager import EndpointRuntimeInterface
import concurrent.futures
import time
import aiohttp, asyncio
from unsync import unsync

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
    total_radix_cache_processing_time: float
    total_internal_request_time: float
    queue_processing_time: float
    tokenization_time: float
    return_time: float
    request_processing_time: float

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
            input_len=input_dict["input_len"],
            total_radix_cache_processing_time=input_dict["total_radix_cache_processing_time"],
            total_internal_request_time=input_dict["total_internal_request_time"],
            queue_processing_time=input_dict["queue_processing_time"],
            tokenization_time=input_dict["tokenization_time"],
            return_time=input_dict["return_time"],
            request_processing_time=input_dict["request_processing_time"]
        )

@dataclass
class LongestPrefixMatchSelector(CustomRuntimeSelector):
    runtimes: List[EndpointRuntimeInterface]
    def __post_init__(self):
        self.metrics_dict = []

    @unsync
    async def get_metrics(self, runtimes, text):
        start = time.time()
        async with aiohttp.ClientSession() as session:
            jobs = [session.post(f"{runtime.url}/scheduling_metrics", json={"prompt": text}) for runtime in runtimes]
            done_jobs = await asyncio.gather(*jobs)
            metrics = [MetricData.from_dict(await done_job.json()) for done_job in done_jobs]
            return metrics

    def runtime_selector(self, text: str):
        # Send a request to each runtime
        start_time = time.time()
        metrics = []
        metrics = self.get_metrics(self.runtimes, text).result()
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
