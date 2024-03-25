import sys
import os
import requests
import concurrent.futures
import math
from argparse import ArgumentParser
from typing import Iterable, List
import numpy as np
import asyncio
import time, datetime
import aiohttp
import logging
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.benchmark_workload_gen import get_react_workload
from multi_node_loader import MultiNodeLoader, ModelDetails

log = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

def async_wrap(f):
    async def _func(*args, **kwargs):
        return f(*args, **kwargs)

    return _func

"""
The workload uses the tokenizer to encode a prompt and then measures the throughput of the tokenizer.
The input prompt is synthetic and may not be representative of the actual workload.
The input is obtained with the react example, the length of generated token id is around 2000.
"""
def profile_tokenizer_throughput(rps: float, t: int, model_details: ModelDetails):
    logging.info("===== Profile Tokenizer Throughput =====")
    logging.info(f'rps={rps}, t={t}')
    num_samples = math.floor(rps * t) if rps != float('inf') else t
    # avoid numeric to ensure consistent token length
    prompts = [(get_react_workload(f'Workload i '), ) for _ in range(num_samples)] 
    tokenizer = model_details.runtimes[0].get_tokenizer()
    
    pool = concurrent.futures.ThreadPoolExecutor()
    
    async def async_encode(text):
        # loop = asyncio.get_running_loop()
        # result = await loop.run_in_executor(pool, tokenizer.encode, text)
        result = await asyncio.to_thread(tokenizer.encode, text)
        return result
    
    start = time.time()
    results = asyncio.run(model_details.async_generate_batch_request_per_sec(
        prompts, rps, 
        async_encode,
        # async_wrap(tokenizer.encode),
    ))
    latency = time.time() - start
    logging.info(f'End to end latency: {latency} seconds')
    
    # def detokenize(ids):
    #     token_texts = tokenizer.convert_ids_to_tokens(ids)
    #     return [t.decode() if isinstance(t, bytes) else t for t in token_texts]
    # for ids in results:
    #     print(detokenize(ids))


def profile_matching_throughput(rps: float, t: int, model_details: ModelDetails):
    logging.info("===== Profile Matching Throughput =====")
    logging.info(f'rps={rps}, t={t}')
    runtime = model_details.runtimes[0]
    metrics_url = f"{runtime.url}/scheduling_metrics"
    
    # Populate radix tree
    num_samples = math.floor(rps * t) if rps != float('inf') else t
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 1
    }
    prompts = [(get_react_workload(f'Workload {i} '), sampling_params) for i in range(num_samples)]
    asyncio.run(model_details.async_generate_batch_request_per_sec(
        prompts, rps, model_details.async_send_request
    ))
    
    # Send metrics request
    async def send_metrics_request(prompt: str):
        async with aiohttp.ClientSession() as session:
            async with session.post(metrics_url, json={"prompt": prompt}) as response:
                metric = await response.json()
                logging.info(metric)
                return metric["matching_overhead"]
            
    prompts = [(get_react_workload(f'Workload {i} '),) for i in range(num_samples)]
    start = time.time()
    overheads = asyncio.run(model_details.async_generate_batch_request_per_sec(
        prompts, rps, send_metrics_request
    ))
    latency = time.time() - start
    logging.info(f'End to end latency: {latency} seconds')
    
   
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="dump_400.log")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.debug(f"Starting Experiment at {datetime.datetime.now(datetime.timezone.utc)}")
    model_name = "mistralai/Mistral-7B-v0.1"
    available_gpus = [0]
    loader = MultiNodeLoader(available_cuda_nodes=available_gpus)
    model_details = loader.load_model(
        model_name, gpus=available_gpus, urls=[]
    )
    configurations_to_profile = [
        # Formats for profile
        # [rps, time(seconds)]
        # OR
        # [inf, batch size]
        
        # [1, 10],
        [float('inf'), 4096],
        # [float('inf'), 200],
        # [float('inf'), 300],
        # [float('inf'), 200],
        # [float('inf'), 400],
    ]
    # profile_task = profile_matching_throughput
    profile_task = profile_tokenizer_throughput
    
    for config in configurations_to_profile:
        profile_task(*config, model_details)
        
    loader.unload_model(model_details)
    