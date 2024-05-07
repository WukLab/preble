from typing import List, Dict, Optional
import numpy as np
import logging
import uuid
from dataclasses import field, asdict, dataclass
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator
import csv
import numpy as np
from enum import Enum, auto
import re

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sglang.srt.managers.router.model_runner import GPUConfig

from benchmarks.benchmark_workload_gen import DataLoader


class ExperimentType(Enum):
    sequential = auto()  # can send the next request only after the previous one is complete
    concurrent_grouped = auto()
    increasing_rps = auto()
    default = auto() # send each one at a fixed rps
    autoscaling = auto()

    def __eq__(self, other):
        return self.value == other.value

@dataclass
class RequestGroup:
    """A group of requests with a specifc request pattern, like sequential"""
    requests: List[Dict]
    request_rate: float
    send_out_times: List[float]
    request_type: ExperimentType


@dataclass
class WorkloadConfig:
    num_prefix_patterns: int
    random_ratio: float
    num_requests: int
    request_rate: float  # the global request rate (may not be the req rate within groups)
    request_groups: List[RequestGroup]
    dataloader: DataLoader
    # send_out_times: List[float]
    exp_time: Optional[float]

    def __repr__(self) -> str:
        return (
            f"=====STARTING BENCHMARK OF {self.num_prefix_patterns} WORKLOADS, "
            f'{self.random_ratio} NON-SHARED, '
            f'{self.num_requests} REQUESTS, '
            f'{self.request_rate} REQ/s, '
            f'{self.exp_time} seconds ====='
        )

@dataclass
class GroupedWorkloadConfig:
    workload_configs: List[WorkloadConfig]

    def __repr__(self) -> str:
        rep_str = ""
        for i, workload_config in enumerate(self.workload_configs):
            rep_str += f"Workload {i}: {workload_config}\n"
        return rep_str

@dataclass
class MajorExperimentArgs:
    workload_configs: Iterator[WorkloadConfig]
    gpu_configs: List[GPUConfig]
    simulate: bool
    log_file_path: str
    selector_configs: List
    model_name: str


@dataclass
class RequestFuncOutput:
    rid: str = ""
    num_gpus: int = 0
    prompt_text: str = ""
    generated_text: str = ""
    success: bool = False
    request_latency: float = 0
    normalized_latency: float = 0
    ttft: float = 0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    global_time: float = 0
    output_len: float = None
    tpot: float = None
    prefill_decode_ratio: float = None
    send_out_time: float = 0.0
    arrival_time: float = 0.0
    append_to_queue_time: float = 0.0
    route_dest: int = None
    scheduling_overhead: float = 0.0
    runtime_selected :int = 0
    max_new_tokens: int = 0

    def update_metrics(
        self,
        tokenizer,
    ):
        # In simulation this will be set
        if self.output_len is None:
            self.output_len = len(tokenizer(self.generated_text).input_ids)
        # print(self.output_len, self.generated_text, self.success, self.error)
        if self.output_len > 1:
            self.tpot = (self.request_latency - self.ttft) / (self.output_len - 1)
        if self.request_latency:
            self.prefill_decode_ratio = self.ttft / self.request_latency
            if self.output_len:
                self.normalized_latency = self.request_latency / self.output_len

    @property
    def total_tokens(self):
        return self.prompt_len + self.output_len

    @property
    def overall_throughput(self):
        return self.total_tokens / self.request_latency

    def to_json(self):
        return json.dumps(self.__dict__)
    

@dataclass
class BenchmarkMetrics:
    num_finished_requests: int
    average_finished_topt: float
    p50_tpot: float
    p90_tpot: float
    p99_tpot: float

    p50_ttft: float
    p90_ttft: float
    p99_ttft: float
    
    p50_norm_latency: float
    p90_norm_latency: float
    p99_norm_latency: float
    avg_norm_latency: float

    ttfts: List[float]
    tpots: List[float]
    throughput_tok_sec: float
    all_results: List[RequestFuncOutput]
    average_request_latency: float
    std_request_latency: float
    average_p90: float
    max_latency: float
    p50_latency: float
    p90_latency: float
    p99_latency: float
    average_ttft: float
    average_topt: float
    prefill_decode_ratio: List[float]
    overall_latency: float
    requests_per_sec: float
    gpu_counts: Dict[int, int]
    avg_scheduling_overhead: float
    max_scheduling_overhead: float
    mean_norm_latency: float

    def gen_benchmark_metrics(
        tokenizer,
        req_func_outputs: List[RequestFuncOutput],
        overall_latency: float,
        time_limit: int = 100,
        gpu_counts={},
        detail_log_path=None,
    ):
        # req_func_outputs = [result for result in req_func_outputs if result.success]
        for result in req_func_outputs:
            result.update_metrics(tokenizer)  # Computes the generated output tokens

        if detail_log_path:
            with open(detail_log_path, "w") as f:
                json.dump([asdict(result) for result in 
                        sorted(req_func_outputs, key=lambda x: x.send_out_time)], 
                        f, indent=4)

        ttfts = [result.ttft for result in req_func_outputs if result.ttft]
        tpots = [result.tpot for result in req_func_outputs if result.tpot]
        overall_latency = overall_latency
        request_latencies = [result.request_latency for result in req_func_outputs if result.request_latency]
        norm_request_latencies = [result.normalized_latency for result in req_func_outputs if result.normalized_latency]
        throughput_tok_sec = (
            sum([result.total_tokens for result in req_func_outputs]) / overall_latency
        )
        all_results = req_func_outputs

        num_finished_requests = sum(
            [
                result.global_time <= time_limit
                for result in req_func_outputs
                if result.success
            ]
        )
        finished_tpot = [
            result.tpot
            for result in req_func_outputs
            if result.tpot is not None
            and result.global_time <= time_limit
            and result.success
        ]
        average_finished_tpot = np.average(finished_tpot)
        p50_tpot, p90_tpot, p99_tpot = np.percentile(tpots, [50, 90, 99])
        
        prefill_decode_ratio = [result.prefill_decode_ratio for result in req_func_outputs if result.prefill_decode_ratio]

        finished_request_latencies = [result.request_latency for result in req_func_outputs if result.success and result.global_time <= time_limit]
        average_request_latency, std_request_latency, average_p90 = (
            np.mean(finished_request_latencies),
            np.std(finished_request_latencies),
            np.percentile(finished_request_latencies, 90),
        )
        max_latency = np.max(finished_request_latencies)
        p50_latency, p90_latency, p99_latency = np.percentile(finished_request_latencies, [50, 90, 99])
        average_ttft = np.mean(ttfts)
        average_topt = np.mean(tpots)
        requests_per_sec = len([req for req in req_func_outputs if req.success]) / overall_latency

        avg_scheduling_overhead = np.mean([result.scheduling_overhead for result in req_func_outputs])
        max_scheduling_overhead = np.max([result.scheduling_overhead for result in req_func_outputs])
        p50_ttft, pt90_ttft, p99_ttft = np.percentile(ttfts, 50), np.percentile(ttfts, 90), np.percentile(ttfts, 99)
        p50_norm_latency, p90_norm_latency, p99_norm_latency = np.percentile(norm_request_latencies, [50, 90, 99])
        mean_norm_latency = np.mean(norm_request_latencies)


        return BenchmarkMetrics(
            num_finished_requests=num_finished_requests,
            average_finished_topt=average_finished_tpot,
            p50_tpot=p50_tpot,
            p90_tpot=p90_tpot,
            p99_tpot=p99_tpot,

            p50_ttft=p50_ttft,
            p90_ttft=pt90_ttft,
            p99_ttft=p99_ttft,
            
            p50_norm_latency=p50_norm_latency,
            p90_norm_latency=p90_norm_latency,
            p99_norm_latency=p99_norm_latency,
            avg_norm_latency=mean_norm_latency,

            ttfts=ttfts,
            tpots=tpots,
            throughput_tok_sec=throughput_tok_sec,
            all_results=all_results,
            average_request_latency=average_request_latency,
            overall_latency=overall_latency,
            std_request_latency=std_request_latency,
            average_p90=average_p90,
            max_latency=max_latency,
            p50_latency=p50_latency,
            p90_latency=p90_latency,
            p99_latency=p99_latency,
            average_ttft=average_ttft,
            average_topt=average_topt,
            prefill_decode_ratio=prefill_decode_ratio,
            requests_per_sec=requests_per_sec,
            gpu_counts=gpu_counts,
            avg_scheduling_overhead=avg_scheduling_overhead,
            max_scheduling_overhead=max_scheduling_overhead,
            mean_norm_latency=mean_norm_latency
        )

    @property
    def num_sucessful_requests(self):
        return sum([1 if result.success else 0 for result in self.all_results])

    def to_json(self):
        all_reqs = [result.to_json() for result in self.all_results]
        return {
            "num_finished_requests": self.num_finished_requests,
            "average_finished_topt": self.average_finished_topt,
            "ttfts": self.ttfts,
            "tpots": self.tpots,
            "overall_latency": self.overall_latency,
            "average_request_latency": self.average_request_latency,
            "std_request_latency": self.std_request_latency,
            "average_p90": self.average_p90,
            "max_latency": self.max_latency,
            "p99_latency": self.p99_latency,
            "average_ttft": self.average_ttft,
            "average_topt": self.average_topt,
            "throughput_tok_sec": self.throughput_tok_sec,
            "all_reqs": all_reqs,
            "prefill_decode_ratio": self.prefill_decode_ratio,
            "scheduling_overhead": self.avg_scheduling_overhead,
        }

    def to_log_file(self, exp_params):
        logging.info(f"Params=({exp_params}) Overall Latency: {self.overall_latency}")
        logging.info(
            f"Params=({exp_params}) Overall Throughput: {self.requests_per_sec}"
        )
        logging.info(
            f"Params=({exp_params}) Overall Request Latency: {self.average_request_latency}, STD: {self.std_request_latency}, P90: {self.average_p90} Norm: {self.mean_norm_latency}"
        )
        logging.info(
            f"Params=({exp_params}) Average TTFT: {self.average_ttft}, Average TOPT: {self.average_topt}, Throughput ToksPerSec: {self.throughput_tok_sec}"
        )
        logging.info(
            f"Params=({exp_params}) Num Finished Requests: {self.num_finished_requests}, Finished Throughput ToksPerSec: {self.average_finished_topt}"
        )
        logging.info(
            f"Params=({exp_params}) Overall Max Latency: {self.max_latency}, P99: {self.p99_latency}"
        )
        logging.info(
            f"Params=({exp_params}) TPOT p50, p90, p99: {self.p50_tpot:.4f}, {self.p90_tpot:.4f}, {self.p99_tpot:.4f}"
        )
        logging.info(
            f"Params=({exp_params}) TTFT p50, p90, p99: {np.percentile(self.ttfts, 50):.4f}, {np.percentile(self.ttfts, 90):.4f}, {np.percentile(self.ttfts, 99):.4f}"
        )
        logging.info(
            f"Params=({exp_params}) Avg Norm Latency: {self.avg_norm_latency:.4f}"
        )
        logging.info(
            f"Params=({exp_params}) Normalized Latency p50, p90, p99: {self.p50_norm_latency:.4f}, {self.p90_norm_latency:.4f}, {self.p99_norm_latency:.4f}"
        )
        logging.info(
            f"Params=({exp_params}) Latency p50, p90, p99: {self.p50_latency:.4f}, {self.p90_latency:.4f}, {self.p99_latency:.4f}"
        )
        logging.info(
            f"Params=({exp_params}) PrefillRatio p50, p90, p99: {np.percentile(self.prefill_decode_ratio, 50):.2f}, {np.percentile(self.prefill_decode_ratio, 90):.2f}, {np.percentile(self.prefill_decode_ratio, 99):.2f}"
        )
        logging.info(
            f"Params=({exp_params}) Overall PrefillRatio: {np.average(self.prefill_decode_ratio)}"
        )
        logging.info(
            f"Params=({exp_params}) Average Scheduling Overhead: {self.avg_scheduling_overhead}"
        )
        logging.info(
            f"Params=({exp_params}) Max Scheduling Overhead: {self.max_scheduling_overhead}"
        )
        logging.info(f"Params=({exp_params}) Counts: {self.gpu_counts}")

    def to_csv_file(self, csv_file, exp_params):
        headers = [
            "policy", "custom_policy", "custom_policy_msg", "rps",  "num_finished_requests", "average_finished_topt", "p50_tpot", "p90_tpot", "p99_tpot",
            "p50_ttft", "p90_ttft", "p99_ttft", "p50_norm_latency", "p90_norm_latency", "p99_norm_latency", "avg_norm_latency", "average_request_latency", "std_request_latency", "average_p90",
            "max_latency", "p50_latency", "p90_latency", "p99_latency", "average_ttft", "average_topt",
            "throughput_tok_sec", "requests_per_sec", "avg_scheduling_overhead", "max_scheduling_overhead", "avg_norm_latency"
        ]
        parsed_params = parse_exp_params(exp_params)
        policy = parsed_params.get("policy", "")
        custom_policy = parsed_params.get("custom_policy", "")
        custom_policy_msg = parsed_params.get("custom_policy_msg", "")

        # Check if file exists to decide whether to write headers
        file_exists = os.path.exists(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)  # Write headers if the file did not exist
            rps = parsed_params.get("rps")
            # Prepare data row
            data = [
                policy, custom_policy, custom_policy_msg, rps,
                self.num_finished_requests, self.average_finished_topt, self.p50_tpot, self.p90_tpot,
                self.p99_tpot, self.p50_ttft, self.p90_ttft, self.p99_ttft, self.p50_norm_latency, self.p90_norm_latency, self.p99_norm_latency, self.avg_norm_latency, self.average_request_latency,
                self.std_request_latency, self.average_p90, self.max_latency, self.p50_latency, self.p90_latency,
                self.p99_latency, self.average_ttft, self.average_topt, self.throughput_tok_sec,
                self.requests_per_sec, self.avg_scheduling_overhead, self.max_scheduling_overhead, self.mean_norm_latency
            ]

            # Write data
            writer.writerow(data)

def parse_exp_params(exp_params):
    """ Parses the experiment parameters to extract specific values. """
    params = {}
    for param in exp_params.split(','):
        key, value = param.split('=')
        params[key.strip()] = value.strip()
    return params