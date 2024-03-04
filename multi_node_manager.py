# %%
from sglang.srt.managers.router.radix_cache import RadixCache
from sglang.utils import get_available_gpu_memory
import socket
from typing import List, DefaultDict
import multiprocessing as mp
import subprocess
import os
import time
import random
import requests
import aiohttp
import os
from collections import defaultdict
from sglang.srt.server import Runtime
import multiprocessing as mp
from enum import Enum
from consistent_hash_trie import ConsistentHashingWithRadixCache

custom_download_dir = "/mnt/ssd1/cache/"

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = custom_download_dir
os.environ["TRANSFORMERS_CACHE"] = custom_download_dir

def get_free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper

class RuntimeSelectionPolicy(Enum):
    RANDOM = 1
    RADIX_CACHE = 2

class ModelDetails:
    def __init__(self, model_path, gpus) -> None:
        self.model_path = model_path
        self.weights = []
        self.runtimes = []
        self.gpus: set = set(gpus)
        self.runtime_selection_policy: RuntimeSelectionPolicy = RuntimeSelectionPolicy.RANDOM
        self.consistent_radix_hash = ConsistentHashingWithRadixCache(num_nodes=len(gpus))

    def get_url(self, port):
        return f"http://localhost:{port}"

    def select_runtime(self, text):
        if self.runtime_selection_policy == RuntimeSelectionPolicy.RANDOM:
            selected_runtime = random.randint(0, len(self.runtimes) - 1)
        elif self.runtime_selection_policy == RuntimeSelectionPolicy.RADIX_CACHE:
            # prefix cache -> consistent hash select which runtime 
            selected_runtime = self.consistent_radix_hash.insert_key(text)
        else:
            raise NotImplementedError
        print(f"Selected runtime {selected_runtime}")
        return self.runtimes[selected_runtime]

    def generate_request(self, text, sampling_params):
        return requests.post(
            self.select_runtime(text).generate_url,
            json={
                "text": text,
                "sampling_params": sampling_params,
            },
            timeout=60,
        )

    async def add_request(self, text, sampling_params):
        # async request
        return await self.select_runtime(text).add_request_await(text, sampling_params)


class MultiNodeLoader:
    def __init__(self, available_cuda_nodes=[]) -> None:
        self.models_allocated = []
        self.gpus_to_model_allocated: DefaultDict[int, List[ModelDetails]] = defaultdict(list)# gpu_id => [model_id, ...]
        self.current_gpu_memory_usage = {}

        # self.current_gpu_memory_usage = {
        #     gpu_id: get_available_gpu_memory(gpu_id, distributed=False)
        #     for gpu_id in available_cuda_nodes
        # }

    def load_model(self, model_path, gpus=[]) -> ModelDetails:
        """
        Load a model onto the specified gpus

        Note: Could manage this directly in python but SGLang uses global variables
        There's also a question on how to unload memory
        """
        model_details = ModelDetails(model_path, gpus)
        self.models_allocated.append(model_details)

        for gpu in gpus:
            runtime = Runtime(model_path=model_path, cuda_devices=[gpu])
            model_details.runtimes.append(runtime)
            self.update_gpu_memory_usage(gpu)
            self.gpus_to_model_allocated[gpu].append(model_details)
        return model_details

    def unload_model(self, model_details: ModelDetails):
        """
        Unload a model from the gpus
        """
        for runtime in model_details.runtimes:
            runtime.shutdown()
        self.models_allocated.remove(model_details)
        for gpu in model_details.gpus:
            self.gpus_to_model_allocated[gpu].remove(model_details)
            self.update_gpu_memory_usage(gpu)
        return model_details

    def update_gpu_memory_usage(self, gpu_id):
        # TODO provide limit to kv cache allocation limit
        # self.current_gpu_memory_usage[gpu_id] = get_available_gpu_memory(
        #     gpu_id, distributed=False
        # )
        pass

# %%
mulit_node_loader = MultiNodeLoader(available_cuda_nodes=[0, 1])
model1 = "mistralai/Mistral-7B-v0.1"
model_details = mulit_node_loader.load_model(model1, gpus=[0, 1])

# %%
from large_string_consts import get_workload1, get_workload2, get_workload

time.time()
print("Testing workload split randomly")

@timeit
def test_model(model_details):
    model_details.generate_request("Workload 1", {})
    model_details.generate_request("Workload 1", {})
    model_details.generate_request("Workload 1", {})

    model_details.generate_request("Workload 2", {})
    model_details.generate_request("Workload 2", {})
    model_details.generate_request("Workload 2", {})
    
    for i in range(100):
        model_details.generate_request(get_workload1(), {})
        model_details.generate_request(get_workload2(), {})

test_model(model_details)

print("Testing workload split based on consistent hash")

model_details.runtime_selection_policy = RuntimeSelectionPolicy.RADIX_CACHE
test_model(model_details)

#%%
# generate workload randomly via asyncio
import asyncio
import time

@timeit
async def test_model(model_details):
    tasks = []
    for i in range(100):
        tasks.append(model_details.add_request(get_workload(i%5), {}))
        tasks.append(model_details.add_request(get_workload(i%5), {}))
    res = await asyncio.gather(*tasks)

async def run_async_tests():
    model_details.runtime_selection_policy = RuntimeSelectionPolicy.RANDOM
    print("Testing Async Policy")
    await test_model(model_details)

    print("Testing Radix Cache Policy")
    model_details.runtime_selection_policy = RuntimeSelectionPolicy.RADIX_CACHE
    await test_model(model_details)

print("Running Async Tests")
# Run Async Tests
asyncio.run(run_async_tests())

# %%


time.sleep(2)
breakpoint()
mulit_node_loader.unload_model(model_details)
print(mulit_node_loader.current_gpu_memory_usage)

# Experiments
    # Try many prefixes
    # Try larger model weights