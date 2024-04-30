from dataclasses import dataclass
from typing import List, Optional, Iterator, Tuple, Dict
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("gpt2")
import sys, os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sglang.srt.managers.router.model_runner import GPUConfig
from enum import Enum, auto
from benchmarks.benchmark_workload_gen import DataLoader
from benchmarks.benchmark_utils import WorkloadConfig
from collections import deque
import numpy as np
import asyncio


class ExperimentType(Enum):
    sequential = auto()  # can send the next request only after the previous one is complete
    concurrent_grouped = auto()
    increasing_rps = auto()
    default = auto() # send each one at a fixed rps
    autoscaling = auto()

    def __eq__(self, other):
        return self.value == other.value

@dataclass
class WorkloadGroupConfig:
    """Extend for a group of requests."""
    rps_type: ExperimentType = ExperimentType.default
    workload: WorkloadConfig = None

class RequestRateManager:
    """Send requests based on the policy of each group of requests."""

    def __init__(self, workloads: List[Tuple[ExperimentType, WorkloadConfig]]):
        self.workloads = []
        for i, workload in enumerate(workloads):
            self.workloads.append(WorkloadGroupConfig(rps_type=workload[0], workload=workload[1]))
        self.ready_requests = deque()  # queue of (workload_id, request)
        self.current_req_done_events: Dict[int, asyncio.Event] = {}  # map from workload_id to asyncio.Event
        self.new_ready_req_event = asyncio.Event()
        self.workload_finished = 0
        self.n_workloads = len(self.workloads)

    async def get_request(self):
        tasks = []
        for i, workload in enumerate(self.workloads):
            task = asyncio.create_task(self.run_req_loop(i, workload))
            tasks.append(task)
        while self.workload_finished < self.n_workloads or len(self.ready_requests) > 0:
            while len(self.ready_requests) > 0:
                yield self.ready_requests.popleft()
            if self.workload_finished == self.n_workloads:
                break
            done, pending = await asyncio.wait([self.new_ready_req_event.wait(),
                                                asyncio.sleep(0.5)], return_when=asyncio.FIRST_COMPLETED)
            self.new_ready_req_event.clear()

    async def run_req_loop(self, workload_id: int, workload_config: WorkloadGroupConfig):
        """Start an async loop for a workload."""
        if workload_config.rps_type == ExperimentType.default:
            await self.send_requests_default(
                workload_id, workload_config.workload.requests, 
                workload_config.workload.request_rate, workload_config.workload.send_out_times)
        elif workload_config.rps_type == ExperimentType.sequential:
            await self.send_requests_sequential(
                workload_id, workload_config.workload.requests, 
                workload_config.workload.request_rate, workload_config.workload.send_out_times)
        else:
            raise NotImplementedError(f"Request policy {workload_config.rps_type} not implemented")
        self.workload_finished += 1
            
    async def send_requests_default(self, 
                                    workload_id: int, 
                                    requests: List[dict], 
                                    request_rate: float, 
                                    send_times: List[float]):
        """Send requests at a fixed rate."""
        input_requests = iter(requests)
        for i, request in enumerate(input_requests):
            self.ready_requests.append((workload_id, request))
            self.new_ready_req_event.set()
            if request_rate == float("inf") and not send_times:
                continue
            if send_times:
                interval = send_times[i + 1] - send_times[i] if i + 1 < len(send_times) else 0
            else:
                interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)

    async def send_requests_sequential(self,
                                       workload_id: int,
                                       requests: List[dict],
                                       request_rate: float,
                                       send_times: List[float]):
        """Send requests sequentially. If the request finishes faster
        than rps, then wait for the next request."""
        self.current_req_done_events[workload_id] = asyncio.Event()
        input_requests = iter(requests)
        for i, request in enumerate(input_requests):
            self.ready_requests.append((workload_id, request))
            self.new_ready_req_event.set()
            if request_rate == float("inf") and not send_times:
                continue
            if send_times:
                interval = send_times[i + 1] - send_times[i] if i + 1 < len(send_times) else 0
            else:
                interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)
            await self.current_req_done_events[workload_id].wait()
            self.current_req_done_events[workload_id].clear()

    def mark_current_req_complete(self, workload_id: int):
        """Internally move to the next requests. 
        This is useful for sequential setting.
        """
        if workload_id in self.current_req_done_events:
            self.current_req_done_events[workload_id].set()

@dataclass
class Workload(WorkloadConfig):
    policy: str
    custom_policy: str
    custom_policy_msg: str
    server_configs: List[GPUConfig]

    def get_starting_policy_message(self):
        custom_msg = ""
        if self.custom_policy_msg:
            custom_msg = f":{self.custom_policy_msg}"
        return f"=====STARTING Policy {self.policy}-{self.custom_policy}{custom_msg}, {self.num_requests} REQUESTS {self.exp_time} seconds {self.workload_params_str()}====="

    def convert_to_custom_format(self, data):
        custom_format = ""
        for key, value in data.items():
            custom_format += f"{key}={value};"
        return custom_format

    def workload_params_str(self) -> str:
        raise NotImplementedError("workload_params_str not implemented")

    def get_tokenizer(self):
        raise NotImplementedError("get_tokenizer not implemented")

@dataclass
class DefaultWorkload(Workload):

    def __repr__(self) -> str:
        return self.dataloader.get_dataset_name()
    
    def workload_params_str(self) -> str:
        workload_args_dict = self.dataloader.workload_specific_args()
        workload_args_dict["rps"] = self.request_rate
        workload_args_dict["policy"] = self.policy
        workload_args_dict["num_requests"] = self.num_requests
        workload_args_dict["policy"] = self.policy.name
        if self.custom_policy:
            workload_args_dict["custom_policy"] = self.custom_policy.name
        workload_args_dict["custom_policy_msg"] = self.custom_policy_msg

        workload_params = ",".join([f"{key}={value}" for key, value in workload_args_dict.items()])
        return workload_params
    
    def get_tokenizer(self):
        return self.dataloader.get_tokenizer()    

@dataclass
class ConfigurableMajorExperimentArgs: 
    log_file_path: str
    csv_log_path: str # for even faster parsing
    simulate: bool
    model_path: str
    experiment_type: ExperimentType
    workload_configs: List[Workload] # Seperate policies/workloads
    experiment_name: str = "basic experiment"

@dataclass
class AllExperiments: # For different experimental runs/like loogle dataset, etc
    experiments: List[ConfigurableMajorExperimentArgs] # For running a batch of experiments
