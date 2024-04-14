from transformers import AutoTokenizer
import random
from benchmark_utils import WorkloadConfig
from benchmark_workload_gen import WorkloadPrefixDataLoader, ToolBenchDataLoader, LooGLEDataset, LooGLEDatasetType
from typing import Iterator
from benchmark_workload_gen import LoadDistribution
import numpy as np
import uuid


def create_workload_prefix_configs(configurations_to_test, model_name, exp_time, num_examples=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for config in configurations_to_test:
        num_workloads, random_ratio, num_requests, request_rate = config
        if exp_time != float("inf"):
            num_requests = int(request_rate * exp_time)
        dataloader = WorkloadPrefixDataLoader(
            num_workloads,
            num_requests,
            tokenizer,
            num_in_context_examples=num_examples,
            output_len=64,
            distribution_of_non_shared=random_ratio,
        )
        requests = dataloader.generate_workload(None)
        random.shuffle(requests)
        send_out_times = [0]
        for i in range(num_requests - 1):
            if request_rate == float('inf'):
                interval = 0
            else:
                interval = np.random.exponential(1 / request_rate)
            send_out_times.append(send_out_times[-1] + interval)
        send_out_times.append(send_out_times[-1]) # for the actual run to calculate last dummy interval
        workload_config = WorkloadConfig(
            num_workloads,
            random_ratio,
            num_requests,
            request_rate,
            requests,
            dataloader,
            send_out_times=send_out_times,
            exp_time=exp_time,
        )
        yield workload_config

def create_toolbench_data_loader(configurations_to_test, model_name, exp_time, data_path, load_dist, k=None) -> Iterator[WorkloadConfig]:
    workload_configs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for config in configurations_to_test:
        num_workloads, num_requests, request_rate = config
        if exp_time != float("inf"):
            num_requests = int(request_rate * exp_time)
        dataloader = ToolBenchDataLoader(
            num_patterns=num_workloads,
            total_num_requests=num_requests,
            tokenizer=tokenizer,
            data_path=data_path,
            load_dist=load_dist,
        )
        requests = dataloader.generate_workload(k=k)
        print(dataloader.load_dist)
        random.shuffle(requests)
        workload_config = WorkloadConfig(
                num_prefix_patterns=num_workloads,
                num_requests=num_requests,
                request_rate=request_rate,
                requests=requests,
                dataloader=dataloader,
                exp_time=exp_time,
                random_ratio=0.0
            )        
        yield workload_config

def create_loogle_dataset(configurations_to_test, model_name, exp_time) -> Iterator[WorkloadConfig]:
    workload_configs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for config in configurations_to_test:
        num_workloads, num_requests, request_rate = config
        if exp_time != float("inf"):
            num_requests = int(request_rate * exp_time)
        dataloader = LooGLEDataset(
            num_patterns=num_workloads,
            total_num_requests=num_requests,
            tokenizer=tokenizer,
            loogle_dataset_type=LooGLEDatasetType.SHORT_QA
        )
        requests = dataloader.generate_workload(max_length=32768)
        random.shuffle(requests)
        workload_config = WorkloadConfig(
                num_prefix_patterns=num_workloads,
                num_requests=num_requests,
                request_rate=request_rate,
                requests=requests,
                dataloader=dataloader,
                exp_time=exp_time,
                random_ratio=0.0
            )        
        yield workload_config

def create_mixture_dataset3(configurations_to_test, model_name, exp_time, data_path, load_dist, k=None) -> Iterator[WorkloadConfig]:
    workload_configs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for config in configurations_to_test:
        num_workloads, num_requests, request_rate = config
        if exp_time != float("inf"):
            num_requests = int(request_rate * exp_time)
        dataloader = ToolBenchDataLoader(
            num_patterns=num_workloads,
            total_num_requests=num_requests,
            tokenizer=tokenizer,
            data_path=data_path,
            load_dist=load_dist,
        )
        toolbench_requests = dataloader.generate_workload(k=k)
        random.shuffle(toolbench_requests)
        dataloader = WorkloadPrefixDataLoader(
            num_workloads,
            num_requests,
            tokenizer,
            num_in_context_examples=4,
            output_len=64,
            distribution_of_non_shared=0.2,
        )
        workload_prefix_requests = dataloader.generate_workload(None)
        random.shuffle(workload_prefix_requests)
        dataloader = ToolBenchDataLoader(
            num_patterns=num_workloads,
            total_num_requests=num_requests,
            tokenizer=tokenizer,
            data_path=data_path,
            load_dist=load_dist,
        )
        toolbench_requests2 = dataloader.generate_workload(k=k)
        random.shuffle(toolbench_requests2)
        workload_config = WorkloadConfig(
                num_prefix_patterns=num_workloads,
                num_requests=num_requests * 3,
                request_rate=request_rate,
                requests=toolbench_requests + workload_prefix_requests +toolbench_requests2,
                dataloader=dataloader,
                exp_time=exp_time * 3,
                random_ratio=0.0
            )        
        yield workload_config


def create_mixture_diff_toolbench_burts(configurations_to_test, model_name, exp_time, data_path,) -> Iterator[WorkloadConfig]:
    workload_configs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for config in configurations_to_test:
        num_workloads, num_requests, request_rate = config
        if exp_time != float("inf"):
            num_requests = int(request_rate * exp_time)
        dataloader = ToolBenchDataLoader(
            num_patterns=num_workloads,
            total_num_requests=num_requests,
            tokenizer=tokenizer,
            data_path=data_path,
            load_dist=LoadDistribution.ZIPF,
        )
        toolbench_requests = dataloader.generate_workload(k=1.1)
        random.shuffle(toolbench_requests)
        dataloader = ToolBenchDataLoader(
            num_patterns=num_workloads,
            total_num_requests=num_requests,
            tokenizer=tokenizer,
            data_path=data_path,
            load_dist=LoadDistribution.NORMAL,
        )
        workload_prefix_requests = dataloader.generate_workload(k=1.1)
        random.shuffle(workload_prefix_requests)
        dataloader = ToolBenchDataLoader(
            num_patterns=num_workloads,
            total_num_requests=num_requests,
            tokenizer=tokenizer,
            data_path=data_path,
            load_dist=LoadDistribution.EVEN,
        )
        toolbench_requests2 = dataloader.generate_workload()
        random.shuffle(toolbench_requests2)
        workload_config = WorkloadConfig(
                num_prefix_patterns=num_workloads,
                num_requests=num_requests * 3,
                request_rate=request_rate,
                requests=toolbench_requests + workload_prefix_requests +toolbench_requests2,
                dataloader=dataloader,
                exp_time=exp_time * 3,
                random_ratio=0.0
            )        
        yield workload_config
