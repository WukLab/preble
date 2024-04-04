import warnings

import sys
import os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from data_parallel_request_cache import (
    DataParallelRuntimeSelectionPolicy,
)
from transformers import AutoTokenizer
from metrics_based_scheduler import LongestPrefixMatchSelector, GlobalLongestPrefixMatch
from benchmark_workload_gen import (
    ToolBenchDataLoader,
    RandomDataLoader,
    LoadDistribution,
    Oracle,
    TBOracle,
    TBOracleB,
    LooGLEDataset,
    LooGLEDatasetType,
    LoogleOracle,
)
from benchmark_utils import BenchmarkMetrics

import random
from model_runtime_manager import GPUConfig

random.seed(10)
import datetime

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
from typing import List, Dict
import paramiko
from model_runtime_manager import RequestFuncOutput

custom_download_dir = "/mnt/ssd1/cache/"

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = custom_download_dir
os.environ["TRANSFORMERS_CACHE"] = custom_download_dir


logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)

log = logging.getLogger(__name__)


class CustomPolicyType(Enum):
    ORACLE = auto()

    TBORACLE = auto()
    TBORACLE_B = auto()

    LPM = auto()
    GLPM = auto()

    LOOGLE_ORACLE = auto()

    LP_SCHEDULER = auto()


def test_oracle_random_basic(
    num_workloads,
    distribution_of_non_shared,
    num_requests,
    rps=0.0,
    exp_time=1800,
    gpu_configs=None,
    model_name="mistralai/Mistral-7B-v0.1",
    load_distribution=LoadDistribution.EVEN,
    k=1.1,
):
    if exp_time != float("inf"):
        num_requests = max(num_requests, int(rps * exp_time))
    loader = MultiNodeLoader()
    logging.debug(
        f"=====STARTING BENCHMARK OF {num_workloads} WORKLOADS, {distribution_of_non_shared} NON-SHARED, {num_requests} REQUESTS, {rps} REQ/s ====="
    )
    logging.debug(f"Using load distribution of {load_distribution}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dataloader = RandomDataLoader(num_workloads, num_requests, tokenizer, LoadDistribution.EVEN, distribution_of_non_shared, 1)
    start_time = time.time()
    # context_length = 4096
    # dataloader = ToolBenchDataLoader(
    #     "G1_workload_updated_input_output_lengths_4096.json",
    #     num_workloads,
    #     num_requests,
    #     tokenizer,
    #     load_dist=load_distribution,
    # )
    # requests = dataloader.generate_workload(k=k)
    dataloader_short = LooGLEDataset(
        loogle_dataset_type=LooGLEDatasetType.SHORT_QA,
        num_patterns=num_workloads,
        total_num_requests=num_requests,
        tokenizer=tokenizer,
        load_dist=LoadDistribution.ALL,
        crop_max_decode=True,
    )
    requests = dataloader_short.generate_workload(max_length=32768)
    random.shuffle(requests)
    print("Data loading time", time.time() - start_time)

    def load_and_run_benchmark(policy, custom_policy=None):
        random.seed(10)
        logging.debug(
            f"=====STARTING Policy {policy}-{custom_policy}, {num_workloads} WORKLOADS, {distribution_of_non_shared} NON-SHARED, {num_requests} REQUESTS, {rps} REQ/s ====="
        )
        model_details = loader.load_model(
            model_name,
            gpu_configs=gpu_configs,
            log_prefix_hit=True,
            # mem_fraction_static=0.42,
            mem_fraction_static=0.8,
            context_length=4096,
        )

        if policy == DataParallelRuntimeSelectionPolicy.CUSTOM:
            if custom_policy == CustomPolicyType.ORACLE:
                oracle = Oracle(
                    num_nodes=len(model_details.runtimes), num_workloads=num_workloads
                )
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=oracle,
                )
            elif custom_policy == CustomPolicyType.TBORACLE:
                oracle = TBOracle(num_nodes=len(model_details.runtimes))
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=oracle,
                )
            elif custom_policy == CustomPolicyType.TBORACLE_B:
                oracle = TBOracleB(num_nodes=len(model_details.runtimes))
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=oracle,
                )
            elif custom_policy == CustomPolicyType.LPM:
                lpm = LongestPrefixMatchSelector(
                    num_nodes=len(model_details.runtimes),
                    runtimes=model_details.runtimes,
                )
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=lpm,
                )
            elif custom_policy == CustomPolicyType.LOOGLE_ORACLE:
                oracle = LoogleOracle(num_nodes=len(model_details.runtimes))
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=oracle,
                )
            elif custom_policy == CustomPolicyType.GLPM:
                glpm = GlobalLongestPrefixMatch(
                    num_nodes=len(model_details.runtimes), model_name=model_name
                )
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=glpm,
                )
            elif custom_policy == CustomPolicyType.LP_SCHEDULER:
                lp_scheduler = LPScheduler(num_nodes=len(model_details.runtimes), depth_limit=4, update_interval=5)
                model_details.update_runtime_selection_policy(
                    DataParallelRuntimeSelectionPolicy.CUSTOM,
                    custom_runtime_selector=lp_scheduler,
                )
        else:
            model_details.update_runtime_selection_policy(policy)

        tic_benchmark = time.time()
        results: List[RequestFuncOutput] = asyncio.run(
            model_details.async_generate_batch_request_per_sec(
                requests,
                rps,
                model_details.async_send_request,
                exp_time,
            )
        )
        overall_latency = time.time() - tic_benchmark
        counts = model_details.request_router.get_model_selection_counts()

        bench_metrics = BenchmarkMetrics.gen_benchmark_metrics(
            tokenizer=tokenizer,
            req_func_outputs=results,
            overall_latency=overall_latency,
            time_limit=exp_time,
            gpu_counts=counts,
        )
        exp_params = f"{model_name}, {num_workloads}, {distribution_of_non_shared}, {num_requests}, {rps}, {policy}-{custom_policy}, {exp_time}"
        bench_metrics.to_log_file(exp_params)

        loader.unload_model(model_details)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)

    # load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.RANDOM, "")
    # load_and_run_benchmark(DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE)
    # load_and_run_benchmark(
    #     DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B
    # )
    load_and_run_benchmark(
        DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.LP_SCHEDULER
    )
    # load_and_run_benchmark(
    #     DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.LOOGLE_ORACLE
    # )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="lp_scheduler_random_scheduler.log")
    # logging.basicConfig(level=logging.DEBUG, filename="experiment_new_benchmarks_4096_toolbench_reasonable_rps.log")
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
        # [200, 0.2, 1024, 50],
        # [ 100, 0.2, 1024, 16],
        [8, 0.2, 200, 0.5],
        # [ 100, 0.2, 4096, 16],
        # [200, 0.2, 4096, 100],
    ]
    gpu_configs = [
        GPUConfig(gpu_id=0, url=None, use_ssh=False),
        GPUConfig(gpu_id=1, url=None, use_ssh=False),
        GPUConfig(
            gpu_id=0,
            url=None,
            use_ssh=True,
            ssh_config={
                "hostname": "192.168.1.18",
                "username": "vikranth",
                "port": 456,
                "python_process": "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python",
                "node_name": "08",
            },
        ),
        GPUConfig(
            gpu_id=1,
            url=None,
            use_ssh=True,
            ssh_config={
                "hostname": "192.168.1.18",
                "username": "vikranth",
                "port": 456,
                "python_process": "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python",
                "node_name": "08",
            },
        ),
    ]

    for config in configurations_to_test:
        test_oracle_random_basic(
            *config,
            exp_time=float("inf"),
            model_name=model_name,
            gpu_configs=gpu_configs,
            load_distribution=LoadDistribution.EVEN,
            k=1.1,
        )
    logging.debug(f"Total Experiment Time: {time.time() - start_time}")
