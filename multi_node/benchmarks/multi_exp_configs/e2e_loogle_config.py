from transformers import AutoTokenizer
import random
import sys, os


# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from multi_experiment_benchmark_utils import AllExperiments, ExperimentType, DefaultWorkload, ConfigurableMajorExperimentArgs

from benchmark_utils import RequestGroup
from benchmark_workload_gen import *
from sglang.srt.managers.router.model_runner import GPUConfig
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy, CustomPolicyType
import random
from multi_exp_configs.multi_exp_utils import *

model_name = "mistralai/Mistral-7B-v0.1"

"""sgalng baseline server runtime config
"""
sglang_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'lpm',
    # 'enable_partial_eviction': True,
    # "chunk_prefill_budget": 512,
}
# GPU Configuration
baseline_gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=2, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=3, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=4, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=5, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=6, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=7, url=None, use_ssh=False, runtime_args=sglang_server_args),
]
add_simulation_to_gpu_config(baseline_gpu_configs)

"""ours server runtime config
"""
ours_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'fcfs-mpq',
    "chunk_prefill_budget": 512,
    'report_hit_ratio': True,
    'enable_iterative_eviction': False,
    'enable_partial_eviction': True,
}
# GPU Configuration
ours_gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=2, url=None, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=3, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=4, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=5, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=6, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=7, url=None, use_ssh=False, runtime_args=ours_server_args),
]
add_simulation_to_gpu_config(ours_gpu_configs)

exp_time = float('inf')
configuration_to_test = [
    # scale_to_gpu([24, 168, 0.3], len(ours_gpu_configs) // 2),
    # scale_to_gpu([24, 281, 0.5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([24, 393, 0.7], len(ours_gpu_configs) // 2),
    # scale_to_gpu([24, 561, 1.0], len(ours_gpu_configs) // 2),
    scale_to_gpu([24, 673, 6], len(ours_gpu_configs) // 2),
]
policies_to_test = [
    # (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs, 'baseline_with_lpm'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerWithoutMissRate, ours_gpu_configs, 'global_without_rebalancing'),
#     (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTime, ours_gpu_configs, 'time_1_6_fresh'),
    (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs, 'baseline_with_lpm'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTimeWithEviction, ours_gpu_configs, ''),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.LOOGLE_ORACLE, baseline_gpu_configs, 'consistent_hashing'),
]

def gen_workloads_for_toolbench(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        num_prefix_patters, num_requests, request_rate = configuration
        dataloader, requests, send_out_times = create_loogle_dataset(
            configuration,
            model_name, 
            exp_time, 
        )
        for policy, custom_policy, server_configs, custom_policy_msg in policies_to_test: # assuming each policy has the exact same settings
            # print(server_configs)
            yield DefaultWorkload(
                    dataloader=dataloader,
                    policy=policy,
                    custom_policy=custom_policy,
                    custom_policy_msg = custom_policy_msg,
                    request_groups=[RequestGroup(requests=requests,
                                                 request_rate=request_rate,
                                                 send_out_times=send_out_times,
                                                 request_type=ExperimentType.default)],
                    # send_out_times=send_out_times,
                    num_prefix_patterns=num_prefix_patters,
                    random_ratio=0.0,
                    exp_time=exp_time,
                    request_rate=request_rate,
                    num_requests=num_requests,
                    server_configs=server_configs,
                )

workloads = gen_workloads_for_toolbench(configuration_to_test, policies_to_test)
loogle_experiment = ConfigurableMajorExperimentArgs(
# <<<<<<< feature/initial_worst_case_benchmarks
    log_file_path="e2e/8r_loogle_rich/exp_8.log",
    csv_log_path="e2e/8r_loogle_rich/exp_8.csv",
    # log_file_path="logs/debug_loogle_cp_2048/exp.log",
    # csv_log_path="logs/debug_loogle_cp_2048/exp.csv",
# =======
#     log_file_path="ckpt_all_in_one_partial_comparison/4r_loogle_k=10/exp.log",
#     csv_log_path="ckpt_all_in_one_partial_comparison/4r_loogle_k=10/exp.csv",
#     # log_file_path="debug/loogle_partial_eviction/exp.log",
#     # csv_log_path="debug/loogle_partial_eviction/exp.csv",
# >>>>>>> mulit_modal_main
    simulate=True,
    model_path=model_name,
    workload_configs=workloads,
    experiment_type=ExperimentType.default,
    experiment_name="loogle_e2e"
)

exp_args = AllExperiments(
    [loogle_experiment]
)