import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import AutoTokenizer
import random
from benchmark_utils import WorkloadConfig
from benchmark_workload_gen import (
    VirtualEnvLoader, 
    WorkloadPrefixDataLoader, 
    ToolBenchDataLoader, 
    LooGLEDataset, 
    LooGLEDatasetType, 
    MultiDomainToolBenchDataLoader,
    ChameleonTabMWPLoader
)
from benchmark_workload_gen import *
from typing import Iterator
from benchmark_workload_gen import LoadDistribution
import numpy as np
import uuid
from benchmarks.exp_configs.model_equations import mistral_7b_A6000_sglang_extend_flashinfer, mistrial_7b_A6000_sglang_decode_flashinfer

ssh_config_08 = {
    "hostname": "192.168.1.18",
    "username": "vikranth",
    "port": 456,
    "python_process": "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python",
    "node_name": "08",
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def scale_to_gpu(workload, gpus):
    new_workload = [w * gpus for w in workload]
    for i in range(2):
        new_workload[i] = int(new_workload[i])
    return new_workload

def pipeline_parallelism(pp: int, forward_equation):
    def pp_forward_time(*args):
        return forward_equation(*args) / pp
    return pp_forward_time


# For Simulator, ignore this if not using it
def add_simulation_to_gpu_config(gpu_configs):
    for config in gpu_configs:
        config.regist_simulator_config(
            [mistral_7b_A6000_sglang_extend_flashinfer, mistrial_7b_A6000_sglang_decode_flashinfer], 
            kv_cache_memory=131072 * 198516,
            lp_forward_simulation=None
        )

def calc_send_out_times(requests, request_rate, exp_time):
    send_out_times = [0]
    for i in range(len(requests) - 1):
        if request_rate == float('inf'):
            interval = 0
        else:
            interval = np.random.exponential(1 / request_rate)
        send_out_times.append(send_out_times[-1] + interval)
    send_out_times.append(send_out_times[-1]) # for the actual run to calculate last dummy interval
    return send_out_times

def create_loogle_dataset(config, model_name, exp_time, max_tokens_override=45) -> Iterator[WorkloadConfig]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_workloads, num_requests, request_rate = config
    if exp_time != float("inf"):
        num_requests = int(request_rate * exp_time)
    print(f"Initialize loogle dataset with {num_workloads} workloads and {num_requests} requests")
    dataloader = LooGLEDataset(
        num_patterns=num_workloads,
        total_num_requests=num_requests,
        tokenizer=tokenizer,
        loogle_dataset_type=LooGLEDatasetType.SHORT_QA,
        max_tokens_override=max_tokens_override
    )
    requests = dataloader.generate_workload(max_length=32768 - max_tokens_override)
    random.shuffle(requests)
    requests = requests[:num_requests]
    print(f"Generated {len(requests)} requests")
    send_out_times = calc_send_out_times(requests, request_rate, exp_time)
    return dataloader, requests, send_out_times

def create_toolbench_dataset(config, model_name, exp_time, data_path, load_dist) -> Iterator[WorkloadConfig]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_workloads, num_requests, request_rate = config
    if exp_time != float("inf"):
        num_requests = int(request_rate * exp_time)
    print(f'Initialize toolbench dataset with {num_workloads} workloads and {num_requests} requests')
    dataloader = ToolBenchDataLoader(
        num_patterns=num_workloads,
        total_num_requests=num_requests,
        tokenizer=tokenizer,
        data_path=data_path,
        load_dist=load_dist,
    )
    requests = dataloader.generate_workload()
    random.shuffle(requests)
    send_out_times = calc_send_out_times(requests, request_rate, exp_time)
    return dataloader, requests, send_out_times

def create_videoQA_dataset(
    config, 
    model_name, 
    exp_time, 
    data_path,
    max_shared_prompt_length
) -> Iterator[WorkloadConfig]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_workloads, num_requests, request_rate = config
    if exp_time != float("inf"):
        num_requests = int(request_rate * exp_time)
    print(f'Initialize VideoQA dataset with {num_workloads} workloads and {num_requests} requests')
    dataloader = VideoDataLoader(
        data_path=data_path,
        total_num_requests=num_requests,
        max_shared_prompt_token_length=max_shared_prompt_length,
        num_patterns=num_workloads,
        tokenizer=tokenizer,
    )
    requests = dataloader.generate_workload()
    random.shuffle(requests)
    send_out_times = calc_send_out_times(requests, request_rate, exp_time)
    return dataloader, requests, send_out_times
        
def create_virtualenv_dataset(config, model_name, exp_time, data_path, load_dist) -> Iterator[WorkloadConfig]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_workloads, num_requests, request_rate = config
    if exp_time != float("inf"):
        num_requests = int(request_rate * exp_time)
    # FIXME: why config is not used
    print(f'Initialize virtualenv dataset')
    dataloader = VirtualEnvLoader(
        tokenizer=tokenizer,
        num_patterns=num_workloads,
        data_path=data_path,
    )
    request_groups = dataloader.generate_workload(k=num_requests)
    random.shuffle(request_groups)
    send_out_times_list = []
    for requests in request_groups:
        # the overall request rate should be split among the different request groups
        send_out_times = calc_send_out_times(requests, request_rate/len(request_groups), exp_time)
        send_out_times_list.append(send_out_times)
    return dataloader, request_groups, send_out_times_list

 
def create_programming_dataset_default(config, model_name, exp_time, max_tokens_override=512) -> Iterator[WorkloadConfig]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_workloads, num_requests, request_rate = config
    if exp_time != float("inf"):
        num_requests = int(request_rate * exp_time)
    print(f'Initialize programing dataset')
    dataloader = ProgrammingDataset(
        num_patterns=num_workloads,
        total_num_requests=num_requests,
        tokenizer=tokenizer,
        max_tokens_override=max_tokens_override,
    )
    requests = dataloader.generate_workload(32768)
    random.shuffle(requests)

    send_out_times = calc_send_out_times(requests, request_rate, exp_time)
    return dataloader, requests, send_out_times

def create_programming_dataset_micro(config, model_name, exp_time, max_tokens_override=512) -> Iterator[WorkloadConfig]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_workloads, num_requests, request_rate = config
    if exp_time != float("inf"):
        num_requests = int(request_rate * exp_time)
    print(f'Initialize programing dataset')
    dataloader = ProgrammingDataset(
        num_patterns=num_workloads,
        total_num_requests=num_requests,
        tokenizer=tokenizer,
        max_tokens_override=max_tokens_override,
        shared_length=2400
    )
    requests = dataloader.generate_workload(32768)
    random.shuffle(requests)

    send_out_times = calc_send_out_times(requests, request_rate, exp_time)
    return dataloader, requests, send_out_times

def create_toolbench_dataset_zipf(config, model_name, exp_time, data_path, load_dist) -> Iterator[WorkloadConfig]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_workloads, num_requests, request_rate, zipf = config
    if exp_time != float("inf"):
        num_requests = int(request_rate * exp_time)
    print(f'Initialize toolbench dataset with {num_workloads} workloads and {num_requests} requests')
    dataloader = ToolBenchDataLoader(
        num_patterns=num_workloads,
        total_num_requests=num_requests,
        tokenizer=tokenizer,
        data_path=data_path,
        load_dist=load_dist,
    )
    requests = dataloader.generate_workload(k=zipf)
    random.shuffle(requests)
    send_out_times = calc_send_out_times(requests, request_rate, exp_time)
    return dataloader, requests, send_out_times