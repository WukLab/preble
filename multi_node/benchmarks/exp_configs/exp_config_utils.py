from transformers import AutoTokenizer
import random
from benchmark_utils import WorkloadConfig
from benchmark_workload_gen import WorkloadPrefixDataLoader, ToolBenchDataLoader, LooGLEDataset, LooGLEDatasetType
from typing import Iterator

def create_workload_prefix_configs(configurations_to_test, model_name, exp_time):
    workload_configs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for config in configurations_to_test:
        num_workloads, random_ratio, num_requests, request_rate = config
        if exp_time != float("inf"):
            num_requests = int(request_rate * exp_time)
        dataloader = WorkloadPrefixDataLoader(
            num_workloads,
            num_requests,
            tokenizer,
            num_in_context_examples=4,
            output_len=64,
            distribution_of_non_shared=random_ratio,
        )
        requests = dataloader.generate_workload(None)
        random.shuffle(requests)
        workload_config = WorkloadConfig(
                num_workloads,
                random_ratio,
                num_requests,
                request_rate,
                requests,
                dataloader,
                exp_time=exp_time,
            )
    yield workload_configs

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
