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
from exp_configs.exp_config_utils import create_workload_prefix_configs, create_toolbench_data_loader, create_loogle_dataset
from exp_configs.react_simulator_config import add_simulation_to_gpu_config

# Basic Configuration
# log_file_path = "logs/sim_hot_cold_rps18_1800.log"
log_file_path = "logs_sim/loogle_non_simulated_4_v2.log"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
exp_time = 100
ssh_config_08 = {
    "hostname": "192.168.1.18",
    "username": "vikranth",
    "port": 456,
    "python_process": "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python",
    "node_name": "08",
}
# GPU Configuration
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=True, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=1, url=None, use_ssh=True, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=0, url=None, use_ssh=False),
    GPUConfig(gpu_id=1, url=None, use_ssh=False),
    # GPUConfig(gpu_id=4, url=None, use_ssh=False),
    # GPUConfig(gpu_id=5, url=None, use_ssh=False),
    # GPUConfig(gpu_id=6, url=None, use_ssh=False),
    # GPUConfig(gpu_id=7, url=None, use_ssh=False),
]
add_simulation_to_gpu_config(gpu_configs)

server_args = {
    "model_path": model_name,
    'gpu_configs': gpu_configs,
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True
}

# Workload Configuration
configurations_to_test = [
    [24, 400, 0.5 * 2],
]
workload_configs = create_loogle_dataset(
    configurations_to_test, 
    model_name, 
    exp_time, 
    # data_path="datasets/G1_workload_updated_input_output_lengths_4096.json", 
    # load_dist=LoadDistribution.EVEN, 
    # k=None
)

# Selector Configuration
# Format {policy - custom policy - message}
selectors_configs = [
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP, 'greedy_no_decode'),
    (DataParallelRuntimeSelectionPolicy.RANDOM, "", "random"),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, "tb_oracle"),
]

exp_args = MajorExperimentArgs(
    server_args,
    workload_configs,
    gpu_configs,
    simulate=False,
    log_file_path=log_file_path,
    selector_configs=selectors_configs,
)
