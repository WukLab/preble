# 3 minute experiments

# 6-7 points with high decode # given a 4k dataset
    # 16, 32, 64, 128, 256, 512, 1024

# Do this for either workload prefix if this works attempt toolbench
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
    "disable_radix_cache": True
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
    'report_hit_ratio': True 
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
    # (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 10),
    # (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 16),
    # (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 32),
    # (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 64),
    # (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 128),
    # (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 256),
    (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 512),
    (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 1024),
    (scale_to_gpu([100, 1000, 10], len(ours_gpu_configs) // 2), 2048),

    # scale_to_gpu([24, 673, 1.2], len(ours_gpu_configs) // 2),
]
policies_to_test = [
    (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs, 'baseline_with_lpm'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerWithoutMissRate, ours_gpu_configs, 'global_without_rebalancing'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTimeWithEviction, ours_gpu_configs, 'time_1_6_fresh'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalScheduler, ours_gpu_configs, 'global_scheduler'),
]

def create_prefix_dataset(config, model_name, exp_time) -> Iterator[WorkloadConfig]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    (num_workloads, num_requests, request_rate), decode_length = config
    if exp_time != float("inf"):
        num_requests = int(request_rate * exp_time)
    print(f"Initialize prefix dataset with {num_workloads} workloads and {num_requests} requests")
    dataloader = WorkloadPrefixDataLoader(
        num_patterns=num_workloads,
        total_num_requests=num_requests,
        tokenizer=tokenizer,
        decoding_size=decode_length,
    )
    requests = dataloader.generate_workload(k=num_requests)
    random.shuffle(requests)
    requests = requests[:num_requests]
    print(f"Generated {len(requests)} requests")
    send_out_times = calc_send_out_times(requests, request_rate, exp_time)
    return dataloader, requests, send_out_times

def gen_workloads_for_toolbench(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        (num_prefix_patters, num_requests, request_rate), decode_length = configuration
        dataloader, requests, send_out_times = create_prefix_dataset(
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
    log_file_path="e2e/8r_prefix_decode_rich/exp_9_1.log",
    csv_log_path="e2e/8r_prefix_decode_rich/exp_9_1.csv",
    # log_file_path="logs/debug_loogle_cp_2048/exp.log",
    # csv_log_path="logs/debug_loogle_cp_2048/exp.csv",
    simulate=False,
    model_path=model_name,
    workload_configs=workloads,
    experiment_type=ExperimentType.default,
    experiment_name="prefix_decode_e2e"
)

exp_args = AllExperiments(
    [loogle_experiment]
)