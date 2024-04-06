from transformers import AutoTokenizer
import random
import sys, os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmark_utils import WorkloadConfig, MajorExperimentArgs
from benchmark_workload_gen import (
    ToolBenchDataLoader,
    RandomDataLoader,
    LoadDistribution,
    Oracle,
    TBOracle,
    TBOracleB,
    LooGLEDataset,
    LooGLEDatasetType,
    LoogleOracle,
)
from sglang.srt.managers.router.model_runner import GPUConfig
from sglang.srt.managers.router.infer_batch import Batch
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy, CustomPolicyType


log_file_path = "logs/debug.log"
model_name = "meta-llama/Llama-2-7b-hf"
configurations_to_test = [
    [200, 0.2, 450, 2.5],
    [100, 0.2, 4096, 4],
]
exp_time = 180
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False),
    GPUConfig(gpu_id=1, url=None, use_ssh=False),
]
server_args = {
    "model_path": model_name,
    'gpu_configs': gpu_configs,
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 2048,
}

#==============================================================================
# For Simulator, ignore this if not using it
def forward_simulation(batch: Batch):
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

for config in gpu_configs:
    config.regist_simulator_config(forward_simulation, 25 << 30) # Llama 2-7b, 0.8 A6000
#==============================================================================

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

selectors_configs = [
    (DataParallelRuntimeSelectionPolicy.RANDOM, None, None),
    (
        DataParallelRuntimeSelectionPolicy.CUSTOM,
        CustomPolicyType.ORACLE,
        Oracle(num_nodes=len(gpu_configs), num_workloads=num_workloads),
    )
]

exp_args = MajorExperimentArgs(
    server_args,
    workload_configs,
    gpu_configs,
    simulate=True,
    log_file_path=log_file_path,
    selector_configs=selectors_configs,
)