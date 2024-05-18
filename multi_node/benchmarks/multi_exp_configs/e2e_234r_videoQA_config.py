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
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'lpm',
    # "chunk_prefill_budget": 512,
    'tp_size': 1,
    'load_format': 'dummy'
}

# GPU Configuration
baseline_gpu_configs = [
    # GPUConfig(gpu_id=0, url='http://0.0.0.0:2333', use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=1, url='http://0.0.0.0:2334', use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=sglang_server_args, ssh_config=ssh_config_06),
    GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=sglang_server_args, ssh_config=ssh_config_06),
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
    'report_hit_ratio': True ,
    'enable_iterative_eviction': False,
    'enable_partial_eviction': False,
    'tp_size': 1,
    'load_format': 'dummy'
}
# GPU Configuration
ours_gpu_configs = [
    # GPUConfig(gpu_id=0, url='http://0.0.0.0:2333', use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=1, url='http://0.0.0.0:2334', use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=ours_server_args, ssh_config=ssh_config_06),
    GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=ours_server_args, ssh_config=ssh_config_06),
    # GPUConfig(gpu_id=2, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=3, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=4, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=5, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=6, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=7, url=None, use_ssh=False, runtime_args=ours_server_args),
]
add_simulation_to_gpu_config(ours_gpu_configs)

def gen_workloads_for_videoQA(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        num_prefix_patters, num_requests, request_rate = configuration
        dataloader, requests, send_out_times = create_videoQA_dataset(
            configuration,
            model_name, 
            exp_time, 
            data_path="datasets/VideoQA.csv",
            max_shared_prompt_length=8192,
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
for i in [2]:
    configuration_to_test = [
        # scale_to_gpu([240, 150, 0.5], i / 2),
        # scale_to_gpu([240, 300, 1], i / 2),
        # scale_to_gpu([240, 600, 2], i / 2),
        scale_to_gpu([150, 900, 3], i / 2),
        # scale_to_gpu([150, 1200, 4], i / 2),
        # scale_to_gpu([150, 1500, 5], i / 2),
        
        # scale_to_gpu([240, 1800, 6], i / 2),
        # scale_to_gpu([240, 2100, 7], i / 2),
        # scale_to_gpu([150, 3000, 10], i / 2),
        # scale_to_gpu([1, 2, 10], i / 2),
    ]
    policies_to_test = [
        # (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs[:i], ''),
        # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.VIDEO_ORACLE, baseline_gpu_configs[:i], ''),
        (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTimeWithEviction, ours_gpu_configs[:i], ''),
    ]
    workloads = gen_workloads_for_videoQA(configuration_to_test, policies_to_test)
    loogle_experiment = ConfigurableMajorExperimentArgs(
        # log_file_path=f"after_merge_upstream/{i}r_videoQA/exp.log",
        # csv_log_path=f"after_merge_upstream/{i}r_videoQA/exp.csv",
        log_file_path="logs/debug_loogle/exp.log",
        csv_log_path="logs/debug_loogle/exp.csv",
        simulate=False,
        model_path=model_name,
        workload_configs=workloads,
        experiment_type=ExperimentType.default,
        experiment_name="videoQA_e2e"
    )
    exp_list.append(loogle_experiment)

exp_args = AllExperiments(
    exp_list
)