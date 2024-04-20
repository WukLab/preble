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
from exp_configs.exp_config_utils import create_multi_domain_toolbench_data_loader
from exp_configs.react_simulator_config import add_simulation_to_gpu_config
import random

# Basic Configuration
# log_file_path = "logs/sim_hot_cold_rps18_1800.log"
log_file_path = "multi_domain_toolbench/multi_domain_test_16_gpus_v2.log"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
exp_time = 200
ssh_config_08 = {
    "hostname": "192.168.1.18",
    "username": "vikranth",
    "port": 456,
    "python_process": "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python",
    "node_name": "08",
}

# GPU Configuration
gpu_configs = [
    # GPUConfig(gpu_id=0, url=None, use_ssh=True, ssh_config=ssh_config_08),
    # GPUConfig(gpu_id=1, url=None, use_ssh=True, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=0, url=None, use_ssh=False, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=1, url=None, use_ssh=False, ssh_config=ssh_config_08),

    GPUConfig(gpu_id=2, url=None, use_ssh=False, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=3, url=None, use_ssh=False, ssh_config=ssh_config_08),

    GPUConfig(gpu_id=4, url=None, use_ssh=False, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=5, url=None, use_ssh=False, ssh_config=ssh_config_08),

    GPUConfig(gpu_id=6, url=None, use_ssh=False, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=7, url=None, use_ssh=False, ssh_config=ssh_config_08),

    GPUConfig(gpu_id=8, url=None, use_ssh=False, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=9, url=None, use_ssh=False, ssh_config=ssh_config_08),

    GPUConfig(gpu_id=10, url=None, use_ssh=False, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=11, url=None, use_ssh=False, ssh_config=ssh_config_08),

    GPUConfig(gpu_id=12, url=None, use_ssh=False, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=13, url=None, use_ssh=False, ssh_config=ssh_config_08),

    GPUConfig(gpu_id=14, url=None, use_ssh=False, ssh_config=ssh_config_08),
    GPUConfig(gpu_id=15, url=None, use_ssh=False, ssh_config=ssh_config_08),


    # GPUConfig(gpu_id=8, url=None, use_ssh=False, ssh_config=ssh_config_08),

    # GPUConfig(gpu_id=2, url=None, use_ssh=False, ssh_config=ssh_config_08),
    # GPUConfig(gpu_id=3, url=None, use_ssh=False, ssh_config=ssh_config_08),

    # GPUConfig(gpu_id=4, url=None, use_ssh=False, ssh_config=ssh_config_08),
    # GPUConfig(gpu_id=5, url=None, use_ssh=False, ssh_config=ssh_config_08),
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
    'context_length': 4096,
    "enable_flashinfer": True,
    # "chunk_prefill_budget": 512
}

# Workload Configuration
# configurations_to_test = [
#     [200, 400, 4],
# ]
configurations_to_test = [
    # [200, 1, 0, 400, 6],
    # [100, 2, 0, 400, 6],
    # [50, 4, 0, 400, 6],
    # [25, 8, 0, 400, 6],

    # [200, 1, 200, 400, 6],
    # [100, 2, 200, 400, 6],
    # [50, 4, 200, 400, 6],
    # [25, 8, 200, 400, 6],

    # [200, 1, 400, 400, 6],
    # [100, 2, 400, 400, 6],
    # [50, 4, 600, 400, 6],
    [200, 16, 600, 400, 6],
    # [200, 8, 600, 400, 6],

    # [200, 400, 12],
]
workload_configs = create_multi_domain_toolbench_data_loader(
    configurations_to_test, 
    model_name, 
    exp_time, 
    data_path="datasets/G1_workload_updated_input_output_lengths_4096.json",
    load_dist=LoadDistribution.EVEN,
    # k = 1.1
)
# Selector Configuration
# Format {policy - custom policy - message}
selectors_configs = [
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.MemSchedulerWithGlobalEviction, 'global_evict'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'tb_oracle_b'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TB_DOMAIN_ORACLE, 'domain_oracle'),
    (DataParallelRuntimeSelectionPolicy.RANDOM, "", 'random'),
]


exp_args = MajorExperimentArgs(
    server_args,
    workload_configs,
    gpu_configs,
    simulate=True,
    log_file_path=log_file_path,
    selector_configs=selectors_configs,
)

if __name__ == "__main__":
    workload_configs = list(workload_configs)
    print(workload_configs[0].requests[0]["text"])
