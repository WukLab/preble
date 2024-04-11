from transformers import AutoTokenizer
import random
import sys, os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmark_utils import WorkloadConfig, MajorExperimentArgs
from benchmark_workload_gen import *
from sglang.srt.managers.router.model_runner import GPUConfig
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy, CustomPolicyType
from exp_configs.exp_config_utils import create_workload_prefix_configs, create_toolbench_data_loader
from exp_configs.react_simulator_config import add_simulation_to_gpu_config

# Basic Configuration
# log_file_path = "logs/sim_hot_cold_rps18_1800.log"
log_file_path = "logs_sim/toolbench_4096_non_simulated.log"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
exp_time = 100

# GPU Configuration
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False),
    GPUConfig(gpu_id=1, url=None, use_ssh=False),
]
add_simulation_to_gpu_config(gpu_configs)

server_args = {
    "model_path": model_name,
    'gpu_configs': gpu_configs,
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 4096,
    "enable_flashinfer": True
}

# Workload Configuration
configurations_to_test = [
    [200, 400, 6],
]
workload_configs = create_toolbench_data_loader(
    configurations_to_test, 
    model_name, 
    exp_time, 
    data_path="datasets/G1_workload_updated_input_output_lengths_4096.json", 
    load_dist=LoadDistribution.EVEN, 
    k=None
)

# Selector Configuration
# Format {policy - custom policy - message}
selectors_configs = [
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP, 'greedy'),
    (DataParallelRuntimeSelectionPolicy.RANDOM, "", "random"),
]

exp_args = MajorExperimentArgs(
    server_args,
    workload_configs,
    gpu_configs,
    simulate=False,
    log_file_path=log_file_path,
    selector_configs=selectors_configs,
)
