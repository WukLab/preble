import warnings

import sys
import os
import pandas as pd
import copy

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpu_stats_profiling import Monitor
import json
import unittest
import time
import numpy as np
import cupy as cp
from data_parallel_request_cache import (
    DataParallelRequestRouter,
    DataParallelRuntimeSelectionPolicy,
    CustomRuntimeSelector,
)
from transformers import AutoTokenizer
from metrics_based_scheduler import LongestPrefixMatchSelector, GlobalLongestPrefixMatch
from parameterized import parameterized
from benchmark_workload_gen import get_react_workload, generate_random_workload, RandomDataLoader, ToolBenchDataLoader, LoadDistribution
import random
random.seed(10)
import datetime
from dataclasses import dataclass
warnings.simplefilter(action="ignore", category=FutureWarning)

from multi_node_loader import MultiNodeLoader
import logging
import asyncio
import torch
from enum import Enum, auto
import gc
import requests as session
import argparse
import re

custom_download_dir = "/mnt/ssd1/cache/"

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = custom_download_dir
os.environ["TRANSFORMERS_CACHE"] = custom_download_dir


logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

class BenchBasicOracleVSRandom(unittest.TestCase):
    @parameterized.expand(
        [
            [2, 0, 1024],
            [10, 0, 1024],
            [100, 0, 1024],
            [200, 0, 1024],
            [2, 0.2, 1024],
            [10, 0.2, 1024],
            [100, 0.2, 1024],
            [200, 0.2, 1024],
            [2, 0.8, 1024],
            [10, 0.8, 1024],
            [100, 0.8, 1024],
            [200, 0.8, 1024],
        ]
    )
    def test_(self):
        pass

class CustomPolicyType(Enum):
    ORACLE = auto()
    LPM = auto()
    GLPM = auto()



@dataclass
class Oracle(CustomRuntimeSelector):
    num_workloads: int
    trace = {}

    def runtime_selector(self, text: str, request_id: str):
        num_nodes = self.num_nodes
        self.trace[request_id] = text[:50]
        for i in range(self.num_workloads):
            if text.startswith(f"Workload {i} "):
                return i % num_nodes
            
        return random.randint(0, num_nodes - 1)
    
@dataclass
class TBOracle(CustomRuntimeSelector):
    trace = {}
    tbl = {}

    def runtime_selector(self, text: str, request_id: str):
        match = re.search(r'You have access of the following tools:\n1.(.+?): ', text)
        if match:
            tool = match.group(1)
            num_nodes = self.num_nodes
            if tool not in self.tbl:
                self.tbl[tool] = random.randint(0, num_nodes - 1)
            return self.tbl[tool]
        else:
            return random.randint(0, self.num_nodes - 1)

def test_oracle_random_basic(num_workloads, distribution_of_non_shared, num_requests, rps=0.0, model_name="mistralai/Mistral-7B-v0.1"):

    loader = MultiNodeLoader(available_cuda_nodes=available_gpus)
    logging.debug(
        f"=====STARTING BENCHMARK OF {num_workloads} WORKLOADS, {distribution_of_non_shared} NON-SHARED, {num_requests} REQUESTS, {rps} REQ/s ====="
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dataloader = RandomDataLoader(num_workloads, num_requests, tokenizer, LoadDistribution.EVEN, distribution_of_non_shared, 1)
    dataloader = ToolBenchDataLoader('/mnt/ssd1/alm-os/ToolBench/data/answer/G1_workload.json', num_workloads, num_requests, tokenizer, LoadDistribution.EVEN)
    requests = dataloader.generate_workload()

    def load_and_run_benchmark(policy, custom_policy=None):
        random.seed(10)
        logging.debug(
            f"=====STARTING Policy {policy}-{custom_policy}, {num_workloads} WORKLOADS, {distribution_of_non_shared} NON-SHARED, {num_requests} REQUESTS, {rps} REQ/s ====="
        )

        model_details = loader.load_model(
            model_name, gpus=available_gpus, urls=[], log_prefix_hit=True,
            # mem_fraction_static=0.42,
            mem_fraction_static=0.8,
        )
        lpm = LongestPrefixMatchSelector(num_nodes=len(available_gpus), runtimes=model_details.runtimes)
        glpm = GlobalLongestPrefixMatch(num_nodes=len(available_gpus), model_name=model_name)
        if policy == DataParallelRuntimeSelectionPolicy.CUSTOM:
            if custom_policy == CustomPolicyType.ORACLE:
                # oracle = Oracle(num_nodes=len(available_gpus), num_workloads=num_workloads)
                oracle = TBOracle(num_nodes=len(available_gpus))
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=oracle,
                )
            elif custom_policy == CustomPolicyType.LPM:
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=lpm,
                )
            elif custom_policy == CustomPolicyType.GLPM:
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=glpm,
                )
        else:
            model_details.update_runtime_selection_policy(policy)
            
        tic_benchmark = time.time()
        if rps > 0.0:
            results = asyncio.run(model_details.async_generate_batch_request_per_sec(
                requests,
                rps,
                model_details.async_send_request,
            ))
        else:
            prompts = [a for a, _ in requests]
            sampling_params = [b for _, b in requests]
            results = model_details.generate_batch_request(
                prompts,
                sampling_params,
                256,
            )
        latency = time.time() - tic_benchmark
        # Each result as a request_latency as a dict. Compute avg, p90 statistics
        request_latencies = [result["request_latency"] for result in results]
        average_request_latency, std_request_latency, average_p90 = np.mean(request_latencies), np.std(request_latencies), np.percentile(request_latencies, 90)
        max_latency, p99_latency = np.max(request_latencies), np.percentile(request_latencies, 99)
        logging.debug(
            f"Params=({model_name}, {num_workloads}, {distribution_of_non_shared}, {num_requests}, {rps}, {policy}-{custom_policy}) Overall Latency: {latency}"
        )
        logging.debug(
            f"Params=({model_name}, {num_workloads}, {distribution_of_non_shared}, {num_requests}, {rps}, {policy}-{custom_policy}) Overall Throughput: {num_requests / latency}"
        )
        logging.debug(
            f"Params=({model_name}, {num_workloads}, {distribution_of_non_shared}, {num_requests}, {rps}, {policy}-{custom_policy}) Overall Request Latency: {average_request_latency}, STD: {std_request_latency}, P90: {average_p90}"
        )
        logging.debug(
            f"Params=({model_name}, {num_workloads}, {distribution_of_non_shared}, {num_requests}, {rps}, {policy}-{custom_policy}) Overall Max Latency: {max_latency}, P99: {p99_latency}"
        )
        # if custom_policy == CustomPolicyType.ORACLE:
        #     with open(f"{exp_name}_metric_{policy}_{custom_policy}_{num_workloads}_{distribution_of_non_shared}_{num_requests}_{rps}.json", "w") as f:
        #         json.dump(oracle.trace, f)
        # if custom_policy == CustomPolicyType.LPM:
        #     with open(f"{exp_name}_metric_{policy}_{custom_policy}_{num_workloads}_{distribution_of_non_shared}_{num_requests}_{rps}.json", "w") as f:
        #         json.dump(lpm.metrics_dict, f)
        # with open(f'{exp_name}_sent_time.json', 'w') as f:
        #     json.dump(model_details.request_sent_time, f)

        df = pd.DataFrame(model_details.request_router.model_selection_stats)
        df.drop("text", axis=1, inplace=True)
        counts = df['selected_runtime'].value_counts().to_dict()
        logging.debug(f"{policy}-{custom_policy}, {counts}")
        
        # for i, runtime in enumerate(model_details.runtimes):
        #     session.post(f'{runtime.url}/dump_prefix_hit_trace', params={'fpath': f'{exp_name}_server_{i}_prefix_hit_trace_{policy}_{custom_policy}_{num_workloads}_{distribution_of_non_shared}_{num_requests}_{rps}.json'})

        loader.unload_model(model_details)
        torch.cuda.empty_cache() 
        gc.collect()
        time.sleep(5)

    load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.RANDOM, "")
    # load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE)
    load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE)
    # load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.LPM)
    # load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GLPM)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="tool_bench.log")
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # Add current time to log file
    start_date = datetime.datetime.utcnow()
    start_time = time.time()
    logging.debug(f"Starting Experiment at {start_date}")
    model_name = "mistralai/Mistral-7B-v0.1"
    # model_name = "lmsys/vicuna-13b-v1.5"
    logging.debug(f"Model Name: {model_name}")
    configurations_to_test = [
        # [200, 0.2, 4096],
        # [10, 0.2, 1024, 5],
        # [100, 0.2, 1024],
        # [200, 0.2, 4096, 0],
        # [300, 0.2, 8192, 0],
        # [200, 0.2, 4096, 0],
        # [250, 0.2, 4096, 100],
        # [200, 0.2, 4096, 5],
        # [200, 0.2, 4096, 75],
        [200, 0.2, 4096, 100],
        # [200, 0.2, 4096, 150],
        # [300, 0.2, 4096, 50],
        # [300, 0.2, 4096, 75],
        # [300, 0.2, 4096, 100],

        # [300, 0.2, 8192, 0],
        # [200, 0.2, 8192, 100],
        # [200, 0.2, 8192, 150],
        # [200, 0.2, 8192, 200],
        # [300, 0.2, 8192, 50],
        # [300, 0.2, 8192, 75],
        # [300, 0.2, 8192, 100],
        # [300, 0.2, 8192, 150],
        # [300, 0.2, 8192, 200],
        # [150, 0.2, 2048, 100],
        # [150, 0.2, 2048, 200],
        # [150, 0.2, 2048, 300],
        # [300, 0.2, 12288, 1024],
        # [300, 0.2, 12288, 2048],
        # [200, 0.8, 1024]
        
        # [200, 0.2, 4096, 20],
        # [200, 0.2, 4096, 100],
        # [300, 0.2, 4096, 50],
        # [300, 0.2, 4096, 100],
        # [10, 0.0, 100, float("inf")],
    ]
    available_gpus = [0, 1]
    exp_name = 'global_selector'
    for config in configurations_to_test:
        test_oracle_random_basic(*config, model_name=model_name)
    logging.debug(f"Total Experiment Time: {time.time() - start_time}")

# 100 random workloads -> 4096 * 0.8 = 3276 shared workloads. 4096 * 0.2 = 812 random workloads 
