# %%
import os
import time
import asyncio
from large_string_consts import get_workload
import os
import concurrent.futures
import time
from gpu_stats_profiling import timeit, async_timeit

from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy
from multi_node_loader import MultiNodeLoader

custom_download_dir = "/mnt/ssd1/cache/"

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = custom_download_dir
os.environ["TRANSFORMERS_CACHE"] = custom_download_dir


# %%
mulit_node_loader = MultiNodeLoader(available_cuda_nodes=[0, 1])
model1 = "mistralai/Mistral-7B-v0.1"
model_details = mulit_node_loader.load_model(
    model1, gpus=[0, 1], urls=[]
)

# %%


@timeit
def test_model(model_details):
    model_details.clear_kv_cache()
    print("Clearing KV Cache") # TODO handle this more determinisitcally
    time.sleep(10)
    def process_requests(workload):
        model_details.generate_request(workload, {})
        model_details.generate_request(workload, {})

    workload_generator = (get_workload(i % 20) for i in range(1000))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1024) as executor:
        executor.map(process_requests, workload_generator)


print("Testing workload split randomly")
model_details.update_runtime_selection_policy(DataParallelRuntimeSelectionPolicy.RANDOM)
test_model(model_details)

print("Testing workload split based on consistent hash")
model_details.runtime_selection_policy = DataParallelRuntimeSelectionPolicy.RADIX_CACHE
test_model(model_details)


# %%
# generate workload randomly via asyncio
@async_timeit
async def test_model(model_details):
    model_details.clear_kv_cache()
    print("Clearing KV Cache") # TODO handle this more determinisitcally
    time.sleep(10)

    tasks = []
    for i in range(2000):
        tasks.append(model_details.async_generate_request(get_workload(i % 4), {}))
        tasks.append(model_details.async_generate_request(get_workload(i % 4), {}))
    await asyncio.gather(*tasks)


async def run_async_tests():
    model_details.runtime_selection_policy = DataParallelRuntimeSelectionPolicy.RANDOM
    print("Testing Random Async Policy")
    await test_model(model_details)

    print("Testing Radix Cache Policy")
    model_details.runtime_selection_policy = DataParallelRuntimeSelectionPolicy.RADIX_CACHE
    await test_model(model_details)


# Run Async Tests
asyncio.run(run_async_tests())

# %%
mulit_node_loader.unload_model(model_details)


# Experiments
# Try many prefixes
# Try larger model weights
