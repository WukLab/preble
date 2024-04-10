from transformers import AutoTokenizer
import random
import sys, os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmark_utils import WorkloadConfig, MajorExperimentArgs
from benchmark_workload_gen import *
from sglang.srt.managers.router.model_runner import GPUConfig
from sglang.srt.managers.router.infer_batch import Batch
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy, CustomPolicyType


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def llama2_7b_A6000_vllm(batch: Batch):
    num_batched_tokens = batch.input_ids.shape[0]
    num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
    
    if num_batched_tokens >= 384:
        forward_time = 0.131*num_batched_tokens + 5.67
    elif num_batched_tokens >= 128:
        forward_time = 0.114*num_batched_tokens + 12.4
    else:
        forward_time = 26.06523603
    forward_time += num_attention_tokens / 2048 * 1.663659159
    forward_time /= 1e3 # to seconds
    return forward_time

def mistral_7b_A6000_sglang(batch: Batch):
    num_batched_tokens = batch.input_ids.shape[0]
    num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
    forward_time = 35
    if num_batched_tokens >= 384:
        forward_time += 0.13*num_batched_tokens - 19.32
    elif num_batched_tokens >= 192:
        forward_time += 0.103*num_batched_tokens - 11.62
    if num_attention_tokens <= 8 * 2048:
        forward_time += 2
    elif num_attention_tokens <= 32 * 2048:
        forward_time += 4
    elif num_attention_tokens <= 48 * 2048:
        forward_time += num_attention_tokens / (48 * 2048) * 14
    else:
        forward_time += num_attention_tokens / (64 * 2048) * 35
    forward_time /= 1e3 # to seconds
    return forward_time

def mistral_7b_A6000_sglang_extend(batch: Batch):
    num_batched_tokens = batch.input_ids.shape[0]
    if num_batched_tokens >= 384:
        forward_time = 0.136 * num_batched_tokens + 10.8
    elif num_batched_tokens >= 192:
        forward_time = 0.103 * num_batched_tokens + 23.5
    else:
        forward_time = 35.0
    return forward_time / 1e3

def mistrial_7b_A6000_sglang_decode(batch: Batch):
    num_batched_tokens = batch.input_ids.shape[0]
    if num_batched_tokens >= 48 * 2048:
        forward_time = num_batched_tokens * 4.5e-4 + 10
    else:
        forward_time = 40.0
    return forward_time / 1e3

def _mistral_7b_A6000_sglang_tp(batch: Batch, k: float):
    forward_time = mistral_7b_A6000_sglang(batch) / k
    return forward_time

def mistral_7b_A6000_sglang_tp(k: float):
    return lambda batch: _mistral_7b_A6000_sglang_tp(batch, k)

# For Simulator, ignore this if not using it
def add_simulation_to_gpu_config(gpu_configs):
    for config in gpu_configs:
        config.regist_simulator_config([mistral_7b_A6000_sglang_extend, mistrial_7b_A6000_sglang_decode], 131072 * 198516)

def create_workload_configs(configurations_to_test):
    workload_configs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for config in configurations_to_test:
        num_workloads, random_ratio, num_requests, request_rate = config
        if exp_time != float("inf"):
            num_requests = int(request_rate * exp_time)
        dataloader = RandomDataLoader(
            num_workloads,
            num_requests,
            tokenizer,
            num_in_context_examples=4,
            output_len=64,
            distribution_of_non_shared=random_ratio,
        )
        requests = dataloader.generate_workload(None)
        random.shuffle(requests)
        workload_configs.append(
            WorkloadConfig(
                num_workloads,
                random_ratio,
                num_requests,
                request_rate,
                requests,
                dataloader,
                exp_time=exp_time,
            )
        )
    return workload_configs

# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------

# Basic Configuration
# log_file_path = "logs/sim_hot_cold_rps18_1800.log"
log_file_path = "logs/log_lp_scheduling_1800.log"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
exp_time = 400

# GPU Configuration
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False),
    GPUConfig(gpu_id=1, url=None, use_ssh=False),
    # GPUConfig(gpu_id=2, url=None, use_ssh=False),
    # GPUConfig(gpu_id=3, url=None, use_ssh=False),
    # GPUConfig(
    #     gpu_id=0,
    #     url=None,
    #     use_ssh=True,
    #     ssh_config={
    #        "hostname": "192.168.1.16",
    #        "username": "wuklab",
    #         "port": 456,
    #         "python_process": "/mnt/data/ssd/sglang_env/bin/python",
    #         "node_name": "06",
    #     },
    # ),
    # GPUConfig(
    #     gpu_id=1,
    #     url=None,
    #     use_ssh=True,
    #     ssh_config={
    #         "hostname": "192.168.1.16",
    #         "username": "wuklab",
    #         "port": 456,
    #         "python_process": "/mnt/data/ssd/sglang_env/bin/python",
    #         "node_name": "06",
    #     },
    # ),
]
add_simulation_to_gpu_config(gpu_configs)

# for config in gpu_configs[2:]:
#     config.regist_simulator_config(mistral_7b_A6000_sglang_tp(2.0), 75 << 30)

# SGLang Runtime Configuration
server_args = {
    "model_path": model_name,
    'gpu_configs': gpu_configs,
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 2048,
}

# Workload Configuration
configurations_to_test = [
    # [200, 0.0, 450, 2.5],
    # [200, 0.0, 4096, 4],
    # [100, 0.0, 4096, 2],
    # [300, 0.0, 4096, 8],
    
    # [300, 0.2, 4096, 8],
    # [300, 0.2, 4096, 12],
    [200, 0.2, 2048, 2],
]
workload_configs = create_workload_configs(configurations_to_test)

# Selector Configuration
# Format {policy - custom policy - message}
selectors_configs = [
    # (DataParallelRuntimeSelectionPolicy.RANDOM, None, '4r'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE, '4r'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE_HOT_COLD, "4r_2h_2c"),
    (DataParallelRuntimeSelectionPolicy.RANDOM, None, "random"),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE, "oracle"),

    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP_GUROBI_SCHEDULER, "greedy"),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.LP_SCHEDULER, "lp"),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.LP_GUROBI_SCHEDULER, "gurobi"),


    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE_HOT_COLD, "3r_2h_1ctp_2.0"),
]

exp_args = MajorExperimentArgs(
    server_args,
    workload_configs,
    gpu_configs,
    simulate=True,
    log_file_path=log_file_path,
    selector_configs=selectors_configs,
)
