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
import asyncio
import requests
from large_string_consts import get_workload1, get_workload2, get_workload
import aiohttp
import os
from collections import defaultdict
from dataclasses import dataclass
from sglang.srt.server import Runtime
import multiprocessing as mp
from enum import Enum
from consistent_hash_trie import ConsistentHashingWithRadixCache
import GPUtil
import concurrent.futures
import time

custom_download_dir = "/mnt/ssd1/cache/"

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = custom_download_dir
os.environ["TRANSFORMERS_CACHE"] = custom_download_dir
from threading import Thread


@dataclass
class GPUStats:
    gpu_id: int
    memory: float
    memoryTotal: float
    load: float
    elapsed_time: float


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.stats = []
        self.start_time = time.time()
        self.start()

    def run(self):
        GPUs = GPUtil.getGPUs()
        while not self.stopped:
            # GPUtil.showUtilization()
            curr_time = time.time() - self.start_time
            for gpu in GPUs:
                self.stats.append(
                    GPUStats(
                        gpu_id=gpu.id,
                        memory=gpu.memoryUtil,
                        memoryTotal=gpu.memoryTotal,
                        load=gpu.load,
                        elapsed_time=curr_time,
                    )
                )
            time.sleep(self.delay)

    def aggregate_stats(self) -> str:
        if len(self.stats) == 0:
            return "No Stats"
        avg_load = sum(stat.load for stat in self.stats) / len(self.stats)
        std_load = sum((stat.load - avg_load) ** 2 for stat in self.stats) / len(
            self.stats
        )
        avg_memory = sum(stat.memory for stat in self.stats) / len(self.stats)
        std_memory = sum((stat.memory - avg_memory) ** 2 for stat in self.stats) / len(
            self.stats
        )
        # Max Load
        max_load = max(stat.load for stat in self.stats)
        max_memory = max(stat.memory for stat in self.stats)
        return f"Load: {avg_load:.2f}±{std_load:.2f}, Memory: {avg_memory:.2f}±{std_memory:.2f}, Max Load: {max_load:.2f}, Max Memory: {max_memory:.2f}"

    def stop(self):
        self.stopped = True


def get_free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        monitor = Monitor(.1)
        result = func(*args, **kwargs)
        end_time = time.time()
        monitor.stop()
        print(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute."
        )
        print(f"Average GPU Stats: {monitor.aggregate_stats()}")
        return result

    return wrapper

def async_timeit(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        monitor = Monitor(.1)
        result = await func(*args, **kwargs)
        end_time = time.time()
        monitor.stop()
        print(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute."
        )
        print(f"Average GPU Stats: {monitor.aggregate_stats()}")
        return result

    return wrapper


class RuntimeSelectionPolicy(Enum):
    RANDOM = 1
    RADIX_CACHE = 2

class URLRuntime:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.generate_url = endpoint + "/generate"
    def shutdown(self):
        pass

class ModelDetails:
    def __init__(self, model_path, gpus) -> None:
        self.model_path = model_path
        self.weights = []
        self.runtimes = []
        self.gpus: set = set(gpus)
        self.runtime_selection_policy: RuntimeSelectionPolicy = (
            RuntimeSelectionPolicy.RANDOM
        )
        self.consistent_radix_hash = ConsistentHashingWithRadixCache(
            num_nodes=len(gpus)
        )
        self.model_selection_stats = []

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
        self.model_selection_stats.append({"selected_runtime": selected_runtime, "text": text, "policy": self.runtime_selection_policy.name})
        # print(f"Selected runtime {selected_runtime}")
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

    def clear_kv_cache(self):
        for runtime in self.runtimes:
            requests.get(runtime.endpoint + "/flush_cache")


    async def async_generate_request(self, text, sampling_params):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                model_details.select_runtime(text).generate_url,
                json={
                    "text": text,
                    "sampling_params": sampling_params,
                },
                timeout=60,
            ) as response:
                return await response.json()

    async def add_request(self, text, sampling_params):
        # async request
        return await self.select_runtime(text).add_request_await(text, sampling_params)


class MultiNodeLoader:
    def __init__(self, available_cuda_nodes=[]) -> None:
        self.models_allocated = []
        self.gpus_to_model_allocated: DefaultDict[int, List[ModelDetails]] = (
            defaultdict(list)
        )  # gpu_id => [model_id, ...]
        self.current_gpu_memory_usage = {}

        # self.current_gpu_memory_usage = {
        #     gpu_id: get_available_gpu_memory(gpu_id, distributed=False)
        #     for gpu_id in available_cuda_nodes
        # }

    def load_model(self, model_path, gpus=[], urls=[]) -> ModelDetails:
        """
        Load a model onto the specified gpus

        Note: Could manage this directly in python but SGLang uses global variables
        There's also a question on how to unload memory
        """
        model_details = ModelDetails(model_path, gpus)
        self.models_allocated.append(model_details)
        def load_runtime(gpu):
            if len(urls) > 0:
                runtime = URLRuntime(urls[gpu])
                model_details.runtimes.append(runtime)
                self.gpus_to_model_allocated[gpu].append(model_details)
            else:
                runtime = Runtime(
                    model_path=model_path,
                    cuda_devices=[gpu],
                    context_length=1024,
                    mem_fraction_static=0.8,
                )
                model_details.runtimes.append(runtime)
            self.update_gpu_memory_usage(gpu)
            self.gpus_to_model_allocated[gpu].append(model_details)
            model_details.gpus.add(gpu)
        
        # parallelizae loading for each gpu
        for gpu in gpus:
            load_runtime(gpu)
        return model_details

    def unload_model(self, model_details: ModelDetails):
        """
        Unload a model from the gpus
        """
        for runtime in model_details.runtimes:
            runtime.shutdown()
        if model_details in self.models_allocated:
            self.models_allocated.remove(model_details)

        for gpu in model_details.gpus:
            self.gpus_to_model_allocated[gpu].remove(model_details)
            self.update_gpu_memory_usage(gpu)
        model_details.runtimes = []
        model_details.gpus = []
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
model_details = mulit_node_loader.load_model(model1, gpus=[0, 1], urls=["http://127.0.0.1:30000", "http://127.0.0.1:3001"])

# %%

@timeit
def test_model(model_details):
    model_details.clear_kv_cache()
    def process_requests(workload):
        model_details.generate_request(workload, {})
        model_details.generate_request(workload, {})
    workload_generator = (get_workload(i % 20) for i in range(1000))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1024) as executor:
        executor.map(process_requests, workload_generator)


print("Testing workload split randomly")
model_details.runtime_selection_policy = RuntimeSelectionPolicy.RANDOM
test_model(model_details)

print("Testing workload split based on consistent hash")
model_details.runtime_selection_policy = RuntimeSelectionPolicy.RADIX_CACHE
test_model(model_details)

import pandas as pd
df = pd.DataFrame(model_details.model_selection_stats)
print(df.drop("text", axis=1).groupby('policy').value_counts())
model_details.model_selection_stats.clear()

# %%
# generate workload randomly via asyncio
@async_timeit
async def test_model(model_details):
    tasks = []
    for i in range(2000):
        tasks.append(model_details.async_generate_request(get_workload(i % 4), {}))
        tasks.append(model_details.async_generate_request(get_workload(i % 4), {}))
    await asyncio.gather(*tasks)

async def run_async_tests():
    model_details.runtime_selection_policy = RuntimeSelectionPolicy.RANDOM
    print("Testing Random Async Policy")
    await test_model(model_details)

    print("Testing Radix Cache Policy")
    model_details.runtime_selection_policy = RuntimeSelectionPolicy.RADIX_CACHE
    await test_model(model_details)

# Run Async Tests
asyncio.run(run_async_tests())

# %%
mulit_node_loader.unload_model(model_details)


# Experiments
# Try many prefixes
# Try larger model weights
# %%

# Requirements added
# -gsputil

# %%
