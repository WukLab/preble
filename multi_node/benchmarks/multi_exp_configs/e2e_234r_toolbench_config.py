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
ssh_config_06 = {
    "hostname": "192.168.1.16",
    "username": "wuklab",
    "port": 456,
    "python_process": "/mnt/data/ssd/zijian_sglang_env/bin/python",
    "node_name": "06",
}

"""sgalng baseline server runtime config
"""
sglang_server_args = {
    'log_prefix_hit': False,
    'mem_fraction_static': 0.8,
    'context_length': 4096,
    "enable_flashinfer": True,
    'schedule_heuristic': 'lpm',
    # "chunk_prefill_budget": 512,
    'load_format': 'dummy',
}

# GPU Configuration
baseline_gpu_configs = [
    # GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=0, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=1,  use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=2,  use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=3,  use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=0, url=None, use_ssh=True, runtime_args=sglang_server_args, ssh_config=ssh_config_06),
    # GPUConfig(gpu_id=1, url=None, use_ssh=True, runtime_args=sglang_server_args, ssh_config=ssh_config_06),
]
add_simulation_to_gpu_config(baseline_gpu_configs)

"""ours server runtime config
"""
ours_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 4096,
    "enable_flashinfer": True,
    'schedule_heuristic': 'fcfs-mpq',
    "chunk_prefill_budget": 1024,
    'report_hit_ratio': True ,
    'enable_iterative_eviction': False,
    'enable_partial_eviction': True,
    'load_format': 'dummy',
}
# GPU Configuration
ours_gpu_configs = [
    # GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=0, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=1,  use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=2,  use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=3,  use_ssh=False, runtime_args=ours_server_args),
]
add_simulation_to_gpu_config(ours_gpu_configs)

def gen_workloads_for_toolbench(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        num_prefix_patters, num_requests, request_rate, _ = configuration
        dataloader, requests, send_out_times = create_toolbench_dataset_zipf(
            configuration,
            model_name, 
            exp_time, 
            data_path="datasets/G1_workload_updated_input_output_lengths_4096.json",
            load_dist=LoadDistribution.ZIPF,
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


exp_time = float('inf')
exp_list = []
for i in [4]:
    configuration_to_test = [
        # scale_to_gpu([320, 900, 3, 1.07], i / 2),
        # scale_to_gpu([320, 1800, 6, 1.07], i / 2),
        # scale_to_gpu([320, 2700, 9, 1.07], i / 2),
        # scale_to_gpu([200, 3600, 12, 1.07], i / 2),
        # scale_to_gpu([400, 3600, 12, 1.07], i / 2),
        # scale_to_gpu([1600, 3600, 12, 1.07], i / 2),
        scale_to_gpu([1000, 3600, 12, 1.1], i / 2),
        # scale_to_gpu([320, 4500, 15, 1.07], i / 2),
        # scale_to_gpu([320, 5400, 18, 1.07], i / 2),
    ]
    policies_to_test = [
        (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs[:i], ''),
        # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.TBORACLE_B, baseline_gpu_configs[:i], 'oracle'),  
        # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTimeWithEviction, ours_gpu_configs[:i], ''),
        # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTimeWithEvictionNoRebalance, ours_gpu_configs[:i], ''),
    ]

    workloads = gen_workloads_for_toolbench(configuration_to_test, policies_to_test)
    loogle_experiment = ConfigurableMajorExperimentArgs(
        log_file_path=f"real_ckpt_all_in_one/rebalancer/rebalancer.log",
        csv_log_path=f"real_ckpt_all_in_one/rebalancer/rebalancer.csv",
        # log_file_path="logs/debug_loogle/exp.log",
        # csv_log_path="logs/debug_loogle/exp.csv",
        simulate=True,
        model_path=model_name,
        workload_configs=workloads,
        experiment_type=ExperimentType.default,
        experiment_name="toolbench_e2e"
    )
    exp_list.append(loogle_experiment)

exp_args = AllExperiments(
    exp_list
)
