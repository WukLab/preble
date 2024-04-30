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
    GPUConfig(gpu_id=4, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=5, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=6, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=7, url=None, use_ssh=False, runtime_args=sglang_server_args),
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
    "chunk_prefill_budget": 512,
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
    GPUConfig(gpu_id=0, url=None, use_ssh=True, runtime_args=ours_server_args),
    GPUConfig(gpu_id=1, url=None, use_ssh=True, runtime_args=ours_server_args),
    GPUConfig(gpu_id=2, url=None, use_ssh=True, runtime_args=ours_server_args),
    GPUConfig(gpu_id=3, url=None, use_ssh=True, runtime_args=ours_server_args),
    GPUConfig(gpu_id=4, url=None, use_ssh=True, runtime_args=ours_server_args),
    GPUConfig(gpu_id=5, url=None, use_ssh=True, runtime_args=ours_server_args),
    GPUConfig(gpu_id=6, url=None, use_ssh=True, runtime_args=ours_server_args),
    GPUConfig(gpu_id=7, url=None, use_ssh=True, runtime_args=ours_server_args),
]
add_simulation_to_gpu_config(ours_gpu_configs)

exp_time = float('inf')
configuration_to_test = [
    # scale_to_gpu([200, 900, 3], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 1800, 6], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 2700, 9], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 3600, 12], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 1000, 15], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 5400, 18], len(ours_gpu_configs) // 2),
    # [200, 7200, 24],
    {
        # 'toolbench': scale_to_gpu([200, 1000, 15], len(ours_gpu_configs) // 2),
        'chameleon': scale_to_gpu([7, 1000, 15], len(ours_gpu_configs) // 2),
    }
]

policies_to_test = [
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerWithoutMissRate, ours_gpu_configs, 'global_scheduler_without_miss_rate'),
    (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", ours_gpu_configs, 'baseline'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTime, ours_gpu_configs, 'global_scheduler'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerWithoutMissRate, ours_gpu_configs, 'global_scheduler_without'),

    # (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs, 'baseline'),
]

def gen_workloads_for_mix(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        requests = []
        request_rate = None
        num_prefix_patters = 0
        for dataset, config in configuration.items():
            assert request_rate is None or request_rate == config[2], "Request rate should be the same for all datasets"
            num_prefix_patters += config[0] if config[0] is not None else 0
            request_rate = config[2]
            if dataset == 'toolbench':
                dataloader, requests_toolbench, _ = create_toolbench_dataset(
                    config,
                    model_name, 
                    exp_time, 
                    data_path="/mnt/data/ssd/dongming/stateful_llm_serving/multi_node/benchmarks/datasets/G1_workload_updated_input_output_lengths_4096.json",
                    load_dist=LoadDistribution.EVEN,
                )
                requests += requests_toolbench
            elif dataset == 'chameleon':
                dataloader, requests_chameleon, _ = create_chameleon_dataset(
                    config,
                    model_name,
                    exp_time,
                    data_path='/mnt/data/ssd/dongming/stateful_llm_serving/chameleon-llm/results/tabmwp/chameleon_gpt4_test_cache.jsonl',
                    load_dist=LoadDistribution.EVEN,
                )
                requests += requests_chameleon
                
        random.shuffle(requests)
        send_out_times = calc_send_out_times(requests, request_rate, exp_time)
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
                    num_requests=len(requests),
                    server_configs=server_configs,
                )

workloads = gen_workloads_for_mix(configuration_to_test, policies_to_test)
toolbench_experiment = ConfigurableMajorExperimentArgs(
    log_file_path="e2e/8r_test_mix_chameleon_exp/exp_v6_🙏.log",
    csv_log_path="e2e/8r_test_mix_chameleon_exp/exp_v6_🙏.csv",
    # log_file_path="logs/debug/exp.log",
    # csv_log_path="logs/debug/exp.csv",
    simulate=True,
    model_path=model_name,
    workload_configs=workloads,
    experiment_type=ExperimentType.default,
    experiment_name="mix_test"
)

exp_args = AllExperiments(
    [toolbench_experiment]
)
