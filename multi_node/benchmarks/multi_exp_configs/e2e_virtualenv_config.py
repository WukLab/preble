from transformers import AutoTokenizer
import random
import sys, os
import copy

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
    # GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=0, url='http://0.0.0.0:2333', use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=1, url='http://0.0.0.0:2334', use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=2, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=3, url=None, use_ssh=False, runtime_args=sglang_server_args),
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
    "chunk_prefill_budget": 512,
    'report_hit_ratio': True,
    "load_format": 'dummy'
}
# GPU Configuration
ours_gpu_configs = [
    # GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=2, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=3, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=0, url='http://0.0.0.0:2333', use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=1, url='http://0.0.0.0:2334', use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=4, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=5, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=6, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=7, url=None, use_ssh=False, runtime_args=sglang_server_args),
]
add_simulation_to_gpu_config(ours_gpu_configs)

exp_time = float('inf')
configuration_to_test = [
    # scale_to_gpu([24, 168, 0.3], len(ours_gpu_configs) // 2),
    # scale_to_gpu([24, 281, 0.5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([24, 393, 0.7], len(ours_gpu_configs) // 2),
    # scale_to_gpu([24, 561, 1.0], len(ours_gpu_configs) // 2),
    # scale_to_gpu([24, 673, 1.2], len(ours_gpu_configs) // 2),
    # scale_to_gpu([100, 200, 5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([230, 2000, 5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([230, 400, 7], len(ours_gpu_configs) // 2),
    # scale_to_gpu([230, 400, 9], len(ours_gpu_configs) // 2),
    # scale_to_gpu([230, 1000, 3], len(ours_gpu_configs) // 2),
    # scale_to_gpu([230, 1000, 5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([230, 1000, 7], len(ours_gpu_configs) // 2),
    # scale_to_gpu([230, 1000, 9], len(ours_gpu_configs) // 2),
    # scale_to_gpu([230, 1000, 12], len(ours_gpu_configs) // 2),

    # scale_to_gpu([200, 900, 3], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 1800, 6], len(ours_gpu_configs) // 2),
    
    # scale_to_gpu([200, 2700, 9], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 3600, 12], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 4500, 15], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 5400, 18], len(ours_gpu_configs) // 2),

    # scale_to_gpu([230, 1000, 9], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 1000, 9], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 1000, 9], len(ours_gpu_configs) // 2),

    # scale_to_gpu([100, 500, 10], len(ours_gpu_configs) // 2),
    # scale_to_gpu([100, 1000, 20], len(ours_gpu_configs) // 2),
    # scale_to_gpu([200, 2000, 9], len(ours_gpu_configs) // 2),

]

policies_to_test = [
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.VirtualenvOracle, baseline_gpu_configs, 'oracle'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTimeWithEviction, ours_gpu_configs, 'all_stuff'),
    # (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs, 'baseline'),

]

def gen_workloads_for_virtualenv(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        num_prefix_patters, num_requests, request_rate = configuration
        dataloader, request_groups, send_out_times = create_virtualenv_dataset_advanced_sequential(
            configuration,
            model_name, 
            exp_time, 
            data_path='/home/exx/nsdi_zijian/stateful_llm_serving/datasets/results_trace_updated_v2.json',
            load_dist=LoadDistribution.EVEN,  # this have no effect on virtualenv
        )
        for policy, custom_policy, server_configs, custom_policy_msg in policies_to_test: 
            req_groups = [RequestGroup(requests=copy.deepcopy(requests),
                                       request_rate=request_rate,  
                                       send_out_times=copy.deepcopy(send_out_times),
                                       request_type=ExperimentType.advanced_sequential) \
                            for requests in request_groups]
            yield DefaultWorkload(
                    dataloader=dataloader,
                    policy=policy,
                    custom_policy=custom_policy,
                    custom_policy_msg = custom_policy_msg,
                    request_groups=req_groups,
                    # send_out_times=send_out_times,
                    num_prefix_patterns=num_prefix_patters,
                    random_ratio=0.0,
                    exp_time=exp_time,
                    request_rate=request_rate,
                    num_requests=num_requests,
                    server_configs=server_configs,
                )

workloads = gen_workloads_for_virtualenv(configuration_to_test, policies_to_test)
loogle_experiment = ConfigurableMajorExperimentArgs(
    log_file_path="real_ckpt_all_in_one/2r_virtualenv_H100/exp.log",
    csv_log_path="real_ckpt_all_in_one/2r_virtualenv_H100/exp.csv",
    simulate=False,
    model_path=model_name,
    workload_configs=workloads,
    experiment_type=ExperimentType.default,
    experiment_name="virtual_e2e"
)

exp_args = AllExperiments(
    [loogle_experiment]
)
