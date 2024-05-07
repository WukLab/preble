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
from multi_exp_configs.multi_exp_utils_dongming import *

model_name = "mistralai/Mistral-7B-v0.1"

"""sgalng baseline server runtime config
"""
sglang_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 4096,
    "enable_flashinfer": True,
    'schedule_heuristic': 'lpm',
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
    'context_length': 4096,
    "enable_flashinfer": True,
    'schedule_heuristic': 'fcfs-mpq',
    # "chunk_prefill_budget": 512,
    'report_hit_ratio': True
}
ssh_config_08 = {
    "hostname": "192.168.1.18",
    "username": "vikranth",
    "port": 456,
    "python_process": "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python",
    "node_name": "08",
}
# GPU Configuration
ours_gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=2, url=None, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=3, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=4, url=None, use_ssh=True, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=5, url=None, use_ssh=True, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=6, url=None, use_ssh=True, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=7, url=None, use_ssh=True, runtime_args=ours_server_args),
]
add_simulation_to_gpu_config(ours_gpu_configs)

exp_time = float('inf')
configuration_to_test = [
    # {
    #     'toolbench': [400, 5400]
    #     'video_qa': [100, 3900, 13]
    # }
    {
        'toolbench': [400, 5400],
        'video_qa': [300, 3900]
    }
]

policies_to_test = [
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerWithoutMissRate, ours_gpu_configs, 'global_scheduler_without_miss_rate'),
    (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs, 'baseline'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalScheduler, ours_gpu_configs, 'global_scheduler'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTime, ours_gpu_configs, 'global_scheduler_with_time'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerWithoutMissRate, ours_gpu_configs, 'global_scheduler_without'),
    # (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs, 'baseline'),
]

def gen_workloads_for_mix(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        requests = []
        num_prefix_patters = 0
        all_requests = []
        for dataset, config in configuration.items():
            num_prefix_patters += config[0] if config[0] is not None else 0
            if dataset == 'toolbench':
                dataloader, requests, send_out_times = create_toolbench_dataset_trace(
                    config,
                    model_name, 
                    exp_time, 
                    data_path="datasets/G1_workload_updated_input_output_lengths_4096.json",
                    load_dist=LoadDistribution.EVEN,
                )
                all_requests.append([requests, send_out_times])
            elif dataset == 'video_qa':
                dataloader, requests, send_out_times = create_videoQA_dataset_trace(
                    config,
                    model_name, 
                    exp_time, 
                    data_path="datasets/VideoQA.csv",
                    max_shared_prompt_length=8192,
                )
                all_requests.append([requests, send_out_times])

        random.shuffle(all_requests)
        for policy, custom_policy, server_configs, custom_policy_msg in policies_to_test: # assuming each policy has the exact same settings
            # print(server_configs)
            yield DefaultWorkload(
                    dataloader=dataloader,
                    policy=policy,
                    custom_policy=custom_policy,
                    custom_policy_msg = custom_policy_msg,
                    request_groups=[RequestGroup(requests=requests,
                                     request_rate=None,
                                     send_out_times=send_out_times,
                                     request_type=ExperimentType.sequential)          
                        for (requests,send_out_times)  in all_requests
                    ],
                    # send_out_times=send_out_times,
                    num_prefix_patterns=num_prefix_patters,
                    random_ratio=0.0,
                    exp_time=exp_time,
                    request_rate=None,
                    num_requests=len(requests),
                    server_configs=server_configs,
                )

workloads = gen_workloads_for_mix(configuration_to_test, policies_to_test)
toolbench_experiment = ConfigurableMajorExperimentArgs(
    log_file_path="e2e_dongming/mixed_azure_trace/4_gpu_random.log",
    csv_log_path="e2e_dongming/mixed_azure_trace/4_gpu_random.csv",
    # log_file_path="logs/debug/exp.log",
    # csv_log_path="logs/debug/exp.csv",
    simulate=False,
    model_path=model_name,
    workload_configs=workloads,
    experiment_type=ExperimentType.default,
    experiment_name="mix_test"
)

exp_args = AllExperiments(
    [toolbench_experiment]
)
