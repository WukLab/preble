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
from exp_configs.model_equations import mistrial_7b_A6000_sglang_total

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def pipeline_parallelism(pp: int, forward_equation):
    def pp_forward_time(*args):
        return forward_equation(*args) / pp
    return pp_forward_time


# For Simulator, ignore this if not using it
def add_simulation_to_gpu_config(gpu_configs):
    for config in gpu_configs:
        config.regist_simulator_config(
            [mistrial_7b_A6000_sglang_total, mistrial_7b_A6000_sglang_total], 
            131072 * 198516)

def create_workload_configs(configurations_to_test):
    workload_configs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for config in configurations_to_test:
        num_workloads, random_ratio, num_requests, request_rate = config
        if exp_time != float("inf"):
            num_requests = int(request_rate * exp_time)
        dataloader = WorkloadPrefixDataLoader(
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
# log_file_path = "logs/more_info_hc_600.log"
# log_file_path = "logs/sim-sleep-flashinfer.log"

# log_file_path = "logs/new_equation_sim_hot_cold_rps18_600/exp.log"
log_file_path = "logs/greedy_new_equation_sim_hot_cold_rps18_600/exp.log"
# log_file_path = "logs/debug.log"

# log_file_path = "logs/sim_hc_debug"

# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
exp_time = 600

# GPU Configuration
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False),
    GPUConfig(gpu_id=1, url=None, use_ssh=False),
    GPUConfig(gpu_id=2, url=None, use_ssh=False),
    GPUConfig(gpu_id=3, url=None, use_ssh=False),
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

# pp_extend = pipeline_parallelism(2, mistral_7b_A6000_sglang_extend)
# pp_decode = pipeline_parallelism(2, mistrial_7b_A6000_sglang_decode)
# gpu_configs[-1].regist_simulator_config([pp_extend, pp_decode], 74 << 30)

# for config in gpu_configs[2:]:
#     config.regist_simulator_config(mistral_7b_A6000_sglang_tp(2.0), 75 << 30)

# SGLang Runtime Configuration
server_args = {
    "model_path": model_name,
    'gpu_configs': gpu_configs,
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 2048,
    'enable_flashinfer': True,
}

# Workload Configuration
configurations_to_test = [
    # [200, 0.0, 450, 2.5],
    # [200, 0.0, 4096, 4],
    # [100, 0.0, 4096, 2],
    # [300, 0.0, 4096, 8],
    
    # [300, 0.2, 4096, 8],
    # [300, 0.2, 4096, 12],
    # [100, 0.2, 4096, 18],
    [200, 0.5, 4096, 18],
]
workload_configs = create_workload_configs(configurations_to_test)

# Selector Configuration
# Format {policy - custom policy - message}
selectors_configs = [
    # (DataParallelRuntimeSelectionPolicy.RANDOM, None, '4r'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE, '4r'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE_HOT_COLD, "4r_2h_2c"),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE_HOT_COLD, "4r_1h_3c"),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE_HOT_COLD, "3r_2h_1ctp_2.0"),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.ORACLE_HOT_COLD, "3r_1hpp_2c"),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GREEDY_LP, 'greedy'),
]

exp_args = MajorExperimentArgs(
    server_args,
    workload_configs,
    gpu_configs,
    simulate=True,
    log_file_path=log_file_path,
    selector_configs=selectors_configs,
)
