from dataclasses import dataclass
from typing import List, Optional, Iterator
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


class ExperimentType(Enum):
    sequential = auto()
    concurrent_grouped = auto()
    increasing_rps = auto()
    default = auto() # send each one at a fixed rps
    autoscaling = auto()

    def __eq__(self, other):
        return self.value == other.value

@dataclass
class Workload(WorkloadConfig):
    policy: str
    custom_policy: str
    custom_policy_msg: str

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
    gpu_configs: List[GPUConfig]
    experiment_type: ExperimentType
    workload_configs: List[Workload] # Seperate policies/workloads
    trace_json_file: Optional[str] = None
    experiment_name: str = "basic experiment"

@dataclass
class AllExperiments: # For different experimental runs/like loogle dataset, etc
    experiments: List[ConfigurableMajorExperimentArgs] # For running a batch of experiments
