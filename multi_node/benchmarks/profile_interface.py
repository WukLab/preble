import sys
import os
import requests
import concurrent.futures
import math
from argparse import ArgumentParser
from typing import Iterable, List
import numpy as np
import asyncio
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.benchmark_workload_gen import get_react_workload
from multi_node_loader import MultiNodeLoader, ModelDetails

"""
The workload uses the tokenizer to encode a prompt and then measures the throughput of the tokenizer.
The input prompt is synthetic and may not be representative of the actual workload.
The input is obtained with the react example, the length of generated token id is around 2000.
"""
def profile_tokenizer_throughput(rps: float, t: int, model_details: ModelDetails):
    print("===== Profile Tokenizer Throughput =====")
    print(f'rps={rps}, t={t}')
    num_samples = math.floor(rps * t) if rps != float('inf') else t
    # avoid numeric to ensure consistent token length
    prompts = [(get_react_workload(f'Workload i '), ) for _ in range(num_samples)] 
    
    async def encode(text):
        return model_details.runtimes[0].get_tokenizer().encode(text)
    start = time.time()
    results = asyncio.run(model_details.async_generate_batch_request_per_sec(
        prompts, rps, encode
    ))
    latency = time.time() - start
    print(f'End to end latency: {latency} seconds')
    
    # def detokenize(ids):
    #     token_texts = tokenizer.convert_ids_to_tokens(ids)
    #     return [t.decode() if isinstance(t, bytes) else t for t in token_texts]
    # for ids in results:
    #     print(detokenize(ids))

def profile_matching_throughput(rps: float, t: int, model_details: ModelDetails):
    pass
    
    
if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-v0.1"
    available_gpus = [0]
    loader = MultiNodeLoader(available_cuda_nodes=available_gpus)
    model_details = loader.load_model(
        model_name, gpus=available_gpus, urls=[]
    )
    configurations_to_profile = [
        # Formats for tokenizer throughput profile
        # [rps, time(seconds)]
        # [inf, batch size]
        # [1, 10],
        [float('inf'), 100],
        [float('inf'), 200],
        [float('inf'), 300],
    ]
    profile_task = profile_tokenizer_throughput
    
    for config in configurations_to_profile:
        profile_task(*config, model_details)
        
    loader.unload_model(model_details)
    