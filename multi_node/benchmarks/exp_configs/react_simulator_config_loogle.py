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
# log_file_path = "eviction_logs_for_load_based_histogram/eviction_load_based_histogram_v5_v2.log"
log_file_path = "perf_logs/loogle_24_393_0.7/exp.log"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
exp_time = float("inf")
ssh_config_08 = {
    "hostname": "192.168.1.18",
    "username": "vikranth",
    "port": 456,
    "python_process": "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python",
    "node_name": "08",
}

server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'fcfs-mpq',
    "chunk_prefill_budget": 2048,
    'report_hit_ratio': True,
}

# GPU Configuration
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=server_args),
    GPUConfig(gpu_id=1, url=None, use_ssh=False,runtime_args=server_args),
]
add_simulation_to_gpu_config(gpu_configs)


configurations_to_test = [
    [ 24, 393, 0.7],
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
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalScheduler, 'fcfs_rebalance_cp2048_hot_cold_load_threshold_1.4'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.BASIC_MEM_SCHEDULERV2, 'mem_basic_v2'),

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
    workload_configs=workload_configs,
    gpu_configs=gpu_configs,
    simulate=True,
    log_file_path=log_file_path,
    selector_configs=selectors_configs,
    model_name=model_name
)

if __name__ == "__main__":
    workload_configs = list(workload_configs)
    print(workload_configs[0].requests[0]["text"])
