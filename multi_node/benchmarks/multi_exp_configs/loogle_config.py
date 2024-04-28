from transformers import AutoTokenizer
import random
import sys, os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from multi_experiment_benchmark_utils import AllExperiments, ExperimentType, DefaultWorkload, ConfigurableMajorExperimentArgs

from benchmark_workload_gen import *
from sglang.srt.managers.router.model_runner import GPUConfig
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy, CustomPolicyType
import random
from multi_exp_configs.multi_exp_utils import create_loogle_dataset, add_simulation_to_gpu_config

model_name = "mistralai/Mistral-7B-v0.1"

sglang_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    "chunk_prefill_budget": 2048,
}
# GPU Configuration
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=sglang_server_args),
]
add_simulation_to_gpu_config(gpu_configs)


exp_time = 300
configuration_to_test = [
    [24, 293, 0.4]
]
policies_to_test = [
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalScheduler, 'load_eviction_v2'),
]

def gen_workloads_for_loogle(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        num_prefix_patters, num_requests, request_rate = configuration
        dataloader, requests, send_out_times = create_loogle_dataset(
            configuration, 
            model_name, 
            exp_time=exp_time,
        )
        for policy, custom_policy, custom_policy_msg in policies_to_test: # assuming each policy has the exact same settings
            yield DefaultWorkload(
                    dataloader=dataloader,
                    policy=policy,
                    custom_policy=custom_policy,
                    custom_policy_msg = custom_policy_msg,
                    requests=requests,
                    send_out_times=send_out_times,
                    num_prefix_patterns=num_prefix_patters,
                    random_ratio=0.0,
                    exp_time=exp_time,
                    request_rate=request_rate,
                    num_requests=num_requests,
                )

workloads = gen_workloads_for_loogle(configuration_to_test, policies_to_test)
loogle_experiment = ConfigurableMajorExperimentArgs(
    log_file_path="eviction_logs_for_load_based_histogram/eviction_load_based_histogram_v5_v2.log",
    csv_log_path="eviction_logs_for_load_based_histogram/eviction_load_based_histogram_v5_v2.csv",
    simulate=True,
    model_path=model_name,
    workload_configs=workloads,
    gpu_configs=gpu_configs,
    experiment_type=ExperimentType.default,
    trace_json_file=None,
    experiment_name="loogle_test"
)

exp_args = AllExperiments(
    [loogle_experiment]
)
