import warnings

import sys
import os
import pandas as pd

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
from parameterized import parameterized
from benchmark_workload_gen import get_react_workload, generate_random_workload
import random
random.seed(10)
import datetime

warnings.simplefilter(action="ignore", category=FutureWarning)

from multi_node_loader import MultiNodeLoader
import logging
import asyncio
import torch
import gc

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


def test_oracle_random_basic(num_workloads, distribution_of_non_shared, num_requests, rps=0.0, model_name="mistralai/Mistral-7B-v0.1"):
    num_prefixed_shared = int(num_requests * (1 - distribution_of_non_shared))
    num_non_shared = int(num_requests * distribution_of_non_shared)
    prompts = []
    for i in range(num_prefixed_shared):
        workload_num = i % num_workloads
        prompts.append(get_react_workload(f"Workload {workload_num} "))
    random_workload = generate_random_workload()
    print(len(prompts))
    for _ in range(num_non_shared):
        prompts.append(random.choice(random_workload))
    random.shuffle(prompts)
    # Save prompts to json dict
    
    class Oracle(CustomRuntimeSelector):
        def runtime_selector(self, text: str):
            num_nodes = self.num_nodes
            for i in range(num_workloads):
                if text.startswith(f"Workload {i} "):
                    return i % num_nodes
                
            return random.randint(0, num_nodes - 1)

    loader = MultiNodeLoader(available_cuda_nodes=available_gpus)
    logging.debug(
        f"=====STARTING BENCHMARK OF {num_workloads} WORKLOADS, {distribution_of_non_shared} NON-SHARED, {num_requests} REQUESTS, {rps} REQ/s ====="
    )

    def load_and_run_benchmark(policy):
        random.seed(10)
        logging.debug(
            f"=====STARTING Policy {policy}, {num_workloads} WORKLOADS, {distribution_of_non_shared} NON-SHARED, {num_requests} REQUESTS, {rps} REQ/s ====="
        )

        model_details = loader.load_model(
            model_name, gpus=available_gpus, urls=[]
        )
        if policy == DataParallelRuntimeSelectionPolicy.CUSTOM:
            model_details.update_runtime_selection_policy(
                DataParallelRuntimeSelectionPolicy.CUSTOM,
                custom_runtime_selector=Oracle(num_nodes=len(available_gpus)),
            )
        else:
            model_details.update_runtime_selection_policy(policy)
        tic_benchmark = time.time()
        if rps > 0.0:
            results = asyncio.run(model_details.async_generate_batch_request_per_sec(
                prompts,
                {
                    "experiment_id": f"random_experiment_{num_workloads}_{distribution_of_non_shared}_{num_requests}",
                    "temperature": 0,
                    "max_new_tokens": 1
                },
                rps,
            ))
        else:
            results = model_details.generate_batch_request(
                prompts,
                {
                    "experiment_id": f"random_experiment_{num_workloads}_{distribution_of_non_shared}_{num_requests}",
                    "temperature": 0,
                    "max_new_tokens": 1
                },
                256,
            )
        latency = time.time() - tic_benchmark
        # Each result as a request_latency as a dict. Compute avg, p90 statistics
        request_latencies = [result["request_latency"] for result in results]
        average_request_latency, std_request_latency, average_p90 = np.mean(request_latencies), np.std(request_latencies), np.percentile(request_latencies, 90)
        
        logging.debug(
            f"Params=({model_name}, {num_workloads}, {distribution_of_non_shared}, {num_requests}, {rps}, {policy}) Overall Latency: {latency}"
        )
        logging.debug(
            f"Params=({model_name}, {num_workloads}, {distribution_of_non_shared}, {num_requests}, {rps}, {policy}) Overall Throughput: {num_requests / latency}"
        )
        logging.debug(
            f"Params=({model_name}, {num_workloads}, {distribution_of_non_shared}, {num_requests}, {rps}, {policy}) Overall Request Latency: {average_request_latency}, STD: {std_request_latency}, P90: {average_p90}"
        )

        df = pd.DataFrame(model_details.request_router.model_selection_stats)
        df.drop("text", axis=1, inplace=True)
        counts = df['selected_runtime'].value_counts().to_dict()
        print(policy, counts)

        loader.unload_model(model_details)
        torch.cuda.empty_cache() 
        gc.collect()
        time.sleep(5)

    load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.RANDOM)
    load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.CUSTOM)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="7b_rps_test.log")
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
        # [200, 0.2, 4096, 5],
        # [200, 0.2, 4096, 10],
        # [200, 0.2, 4096, 20],
        # [200, 0.2, 4096, 30],
        
        # [200, 0.2, 8192, 50],
        # [200, 0.2, 8192, 100],
        # [200, 0.2, 8192, 150],
        # [200, 0.2, 8192, 200],
        # [300, 0.2, 8192, 50],
        # [300, 0.2, 8192, 75],
        [300, 0.2, 8192, 100],
        # [300, 0.2, 8192, 150],
        # [300, 0.2, 8192, 200],
        
        # [150, 0.2, 2048, 100],
        # [150, 0.2, 2048, 200],
        # [150, 0.2, 2048, 300],
        # [300, 0.2, 12288, 1024],
        # [300, 0.2, 12288, 2048],
        # [200, 0.8, 1024]
    ]
    available_gpus = [0, 1]
    for config in configurations_to_test:
        test_oracle_random_basic(*config, model_name=model_name)
    logging.debug(f"Total Experiment Time: {time.time() - start_time}")

# 100 random workloads -> 4096 * 0.8 = 3276 shared workloads. 4096 * 0.2 = 812 random workloads 
