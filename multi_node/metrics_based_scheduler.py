from dataclasses import dataclass
from typing import List
import random
from data_parallel_request_cache import CustomRuntimeSelector
from model_runtime_manager import EndpointRuntimeInterface
from sglang.srt.managers.router.radix_cache import RadixCache
import transformers
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
    routing_time: float
    waiting_time: float
    inner_router_time: float
    tokenizer_manager_waiting_time: float
    manager_tokenizer_waiting_time: float
    manager_recv_time: float

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
            request_processing_time=input_dict["request_processing_time"],
            routing_time=input_dict["routing_time"],
            waiting_time=input_dict["waiting_time"],
            inner_router_time=input_dict["inner_router_time"],
            tokenizer_manager_waiting_time=input_dict['tokenizer_manager_waiting_time'],
            manager_tokenizer_waiting_time=input_dict['manager_tokenizer_waiting_time'],
            manager_recv_time=input_dict['manager_recv_time']
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
    
    def fair_random(self, text: str):
        if not text.startswith("Workload"):
            return random.randint(0, len(self.runtimes) - 1)
        else:
            return int(text.split(" ")[1]) % len(self.runtimes)

    def runtime_selector(self, text: str, request_id: str):
        # Send a request to each runtime
        start_time = time.time()
        metrics = []
        metrics = self.get_metrics(self.runtimes, text).result()
        # Handle the case where the prefix match len/input length is really small less than 1%. Pick randomly
        if all(metric.prefix_match_len / metric.input_len < 0.02 for metric in metrics):
            selected_runtime = self.fair_random(text[:20])
        elif all(metric.prefix_match_len / metric.input_len >= 0.5 for metric in metrics):
            selected_runtime = self.fair_random(text[:20])
        else:
            # Pick the runtime with the highest prefix match len/input length
            max_prefix_match = max(range(len(self.runtimes)), key=lambda i: metrics[i].prefix_match_len / metrics[i].input_len)
            selected_runtime = max_prefix_match
        
        self.metrics_dict.append({
            "text": text[:100],
            "rid": request_id,
            "metrics": [metric.__dict__ for metric in metrics],
            "selected_runtime": selected_runtime,
            "overhead": time.time() - start_time
        })
            
        return selected_runtime


class GlobalLongestPrefixMatch(CustomRuntimeSelector):
    def __init__(self, num_nodes: int, model_name: str):
        self.num_nodes = num_nodes
        self.tree_caches = [
            RadixCache() for _ in range(num_nodes)
        ]

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.waiting_queues = [0 for _ in range(num_nodes)]
        self.metrics_dict = []

    def runtime_selector(self, text: str, request_id: str):
        # Tokenize the text
        start_time = time.time()
        tokens = self.tokenizer.encode(text)[:1024 - 1]
        # Find the longest prefix match
        prefix_match_length = [self.tree_caches[i].match_prefix(tokens)[0] for i in range(self.num_nodes)]
        percent_matched = [len(match) / len(tokens) for match in prefix_match_length]

        if all(match_percent< 0.02 for match_percent in percent_matched):
            runtime_selected = random.randint(0, self.num_nodes - 1)
        else:
            runtime_selected = max(range(self.num_nodes), key=lambda i: (percent_matched[i], -self.waiting_queues[i]))

        # Insert the tokenized text into the radix cache
        self.tree_caches[runtime_selected].insert(tuple(tokens))
        self.waiting_queues[runtime_selected] += 1

        self.metrics_dict.append({
            "text": text[:15],
            "rid": request_id,
            "selected_runtime": runtime_selected,
            "overhead": time.time() - start_time
        })
        return runtime_selected
