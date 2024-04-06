import warnings
import random
import numpy as np
random.seed(10)
np.random.seed(10)

import sys
import os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time

import datetime

warnings.simplefilter(action="ignore", category=FutureWarning)

from multi_node_loader import MultiNodeLoader
from model_runtime_manager import RequestFuncOutput, ModelDetails
from benchmark_utils import BenchmarkMetrics, MajorExperimentArgs, WorkloadConfig
from benchmarks.exp_configs.react_simulator_config import exp_args
from benchmark_workload_gen import *
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy, CustomPolicyType
from metrics_based_scheduler import LongestPrefixMatchSelector, GlobalLongestPrefixMatch
from global_policy_lp import LPScheduler

import logging
import torch
import gc
import requests as session
import argparse
import re
from typing import List, Dict
import paramiko
import importlib.util

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

def regist_selector(policy, custom_policy, model_details: ModelDetails, workload_config: WorkloadConfig):
    if policy == DataParallelRuntimeSelectionPolicy.CUSTOM:
        if custom_policy == CustomPolicyType.ORACLE:
            oracle = Oracle(
                num_nodes=len(model_details.runtimes), 
                num_workloads=workload_config.num_prefix_patterns,
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

def load_and_run_benchmark(
    model_details: ModelDetails, 
    workload_config: WorkloadConfig, 
    policy, custom_policy=None,
):
    num_workloads = workload_config.num_prefix_patterns
    distribution_of_non_shared = workload_config.random_ratio
    num_requests = workload_config.num_requests
    rps = workload_config.request_rate
    exp_time = workload_config.exp_time
    requests = workload_config.requests
    tokenizer = workload_config.dataloader.tokenizer
        
    logging.debug(
        f"=====STARTING Policy {policy}-{custom_policy}, {num_workloads} WORKLOADS, {distribution_of_non_shared} NON-SHARED, {num_requests} REQUESTS, {rps} REQ/s, {exp_time} seconds ====="
    )
    regist_selector(policy, custom_policy, model_details, workload_config)
    
    tic_benchmark = time.time()
    results: List[RequestFuncOutput] = model_details.get_experiment_results(
        requests, rps, exp_time
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


def test_oracle_random_basic(exp_args: MajorExperimentArgs):
    loader = MultiNodeLoader(exp_args.simulate)
    for workload_config in exp_args.workload_configs:
        logging.debug(workload_config)
        logging.debug(f"Using load distribution of {workload_config.dataloader.load_dist}")
        # dataloader = RandomDataLoader(num_workloads, num_requests, tokenizer, LoadDistribution.EVEN, distribution_of_non_shared, 1)
        for selector_config in exp_args.selector_configs:
            model_details = loader.load_model(**exp_args.runtime_args) # TODO: clear cache instead of reload
            load_and_run_benchmark(model_details, workload_config, *selector_config)
            loader.unload_model(model_details)
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename=exp_args.log_file_path)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # Add current time to log file
    start_date = datetime.datetime.utcnow()
    start_time = time.time()
    logging.debug(f"Starting Experiment at {start_date}")
    # model_name = "mistralai/Mistral-7B-v0.1"
    # model_name = "lmsys/vicuna-13b-v1.5"
    model_name = exp_args.runtime_args['model_path']
    logging.debug(f"Model Name: {model_name}")

    test_oracle_random_basic(exp_args)
    logging.debug(f"Total Experiment Time: {time.time() - start_time}")
