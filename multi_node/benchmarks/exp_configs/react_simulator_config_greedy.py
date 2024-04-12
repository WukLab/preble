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
from exp_configs.exp_config_utils import create_workload_prefix_configs
from exp_configs.react_simulator_config import add_simulation_to_gpu_config

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------

# Basic Configuration
# log_file_path = "logs/sim_hot_cold_rps18_1800.log"
log_file_path = "logs/test_greedy_simulation/exp.log"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
exp_time = 100

# GPU Configuration
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False),
    GPUConfig(gpu_id=1, url=None, use_ssh=False),
]
add_simulation_to_gpu_config(gpu_configs)

# for config in gpu_configs[2:]:
#     config.regist_simulator_config(mistral_7b_A6000_sglang_tp(2.0), 75 << 30)

# SGLang Runtime Configuration
server_args = {
    "model_path": model_name,
    'gpu_configs': gpu_configs,
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 2048,
    "enable_flashinfer": True
}

# Workload Configuration
configurations_to_test = [
    [200, 0.2, 400, 4],
]
workload_configs = create_workload_prefix_configs(configurations_to_test, model_name, exp_time)

# Selector Configuration
# Format {policy - custom policy - message}
selectors_configs = [
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP, 'greedy'),
    (DataParallelRuntimeSelectionPolicy.RANDOM, "", "random"),
]

exp_args = MajorExperimentArgs(
    server_args,
    workload_configs,
    gpu_configs,
    simulate=True,
    log_file_path=log_file_path,
    selector_configs=selectors_configs,
)
