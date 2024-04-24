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
from exp_configs.exp_config_utils import create_workload_prefix_configs, create_toolbench_data_loader, create_loogle_dataset, create_mixture_diff_toolbench_burts
from exp_configs.react_simulator_config import add_simulation_to_gpu_config
import random

# Basic Configuration
# log_file_path = "logs/sim_hot_cold_rps18_1800.log"
log_file_path = "eviction_logs_for_load_based_histogram/eviction_load_based_histogram_v5_v2.log"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
exp_time = 300
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
    "enable_flashinfer": True,
    "chunk_prefill_budget": 2048,
}

# Workload Configuration
# configurations_to_test = [
#     [200, 400, 4],
# ]s
configurations_to_test = [
    [ 24, 393, 0.4],
    # [ 24, 393, 0.5]
]
workload_configs = create_loogle_dataset(
    configurations_to_test, 
    model_name, 
    exp_time, 
    # data_path="datasets/G1_workload_updated_input_output_lengths_4096.json",
    # load_dist=LoadDistribution.ZIPF,
    # k=1.05
    # k=None
)

# Selector Configuration
# Format {policy - custom policy - message}
selectors_configs = [
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP_OLD, 'greedy_old'),
# (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HiostgramBasedRecompLoad, 'recomp_load'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.MemSchedulerEvictBasedOnLoad, 'evict_based_on_load'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HistogramBasedMemoryLoadScheduler, 'load'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HiostgramBasedRecompLoad, 'recomp'),
    # (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", 'round_robin'),
    # (DataParallelRuntimeSelectionPolicy.RANDOM, "", 'random'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HiostgramBasedRecompLoadWithEvictionV2, 'load_eviction_v2'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.BASIC_MEM_SCHEDULERV2, 'mem_basic_v2'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HistogramBasedMemoryLoadScheduler, 'load_scheduler'),
        # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HiostgramBasedRecompLoadWithEviction, 'recomp_scheduler_with_eviction'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HiostgramBasedRecompLoad, 'recomp_scheduler_without_eviction'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HiostgramBasedRecompLoad, 'recomp_scheduler'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HiostgramBasedRecompLoadWithEviction, 'recomp_scheduler_with_eviction'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.HiostgramBasedRecompLoad, 'recomp_scheduler'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.MemSchedulerEvictBasedOnLoad, 'evict_based_on_load'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.BASIC_MEM_SCHEDULER, 'greedy_v3'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.BASIC_MEM_SCHEDULER, 'greedy_v3'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.LOOGLE_ORACLE, 'loogle_oracle'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.LOOGLE_ORACLE, 'loogle_oracle'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'tb_oracle_b'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'tb_oracle_b'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'tb_oracle_b'),
    # # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP, 'greedy_v3'),

    # # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'tb_oracle_b'),
    # # # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'tb_oracle_b'),

    # (DataParallelRuntimeSelectionPolicy.RANDOM, "", 'random'),
    # (DataParallelRuntimeSelectionPolicy.RANDOM, "", 'random'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP, 'greedy_v3'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'oracle'),
    # (DataParallelRuntimeSelectionPolicy.RANDOM, "", 'random'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP_OLD, 'greedy_old'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP, 'greedy'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'oracle'),
    # (DataParallelRuntimeSelectionPolicy.RANDOM, "", 'random'),

    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP_OLD, 'greedy_old'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP, 'greedy'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, 'oracle'),
    # (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", 'round_robin'),
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
