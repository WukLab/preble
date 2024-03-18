import requests
from data_parallel_request_cache import (
    DataParallelRuntimeSelectionPolicy,
    ConsistentHashingWithRadixCache,
)
from data_parallel_request_cache import DataParallelRequestRouter
import aiohttp
import uuid
from dataclasses import dataclass
from sglang.srt.server import Runtime as SGLangServer
from typing import List, Iterable
from concurrent.futures import ThreadPoolExecutor
import json
import asyncio
import numpy as np
import time

@dataclass
class EndpointRuntimeInterface:
    def __post_init__(self):
        self.runtime_id = str(uuid.uuid4())
        assert self.url is not None
        self._generate_url = f"{self.url}/generate"

    @property
    def generate_url(self):
        return self._generate_url

    @generate_url.setter
    def generate_url(self, url):
        self._generate_url = url

    @property
    def flush_cache_url(self):
        return f"{self.url}/flush_cache"

    def shutdown(self):
        pass

class URLRuntime(EndpointRuntimeInterface):
    def __init__(self, url, gpu):
        super().__init__()
        self.url = url
        self.gpu = gpu

class ExtendedSGLangRuntime(SGLangServer, EndpointRuntimeInterface):
    def __init__(self, gpu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu = gpu


def random_uuid_string():
    return str(uuid.uuid4())


class ModelDetails:
    """
    Supports Data Parallel Model Allocation
    """

    def __init__(
        self, model_path, gpus, runtime_selection_policy=DataParallelRuntimeSelectionPolicy.RANDOM
    ) -> None:
        self.model_path = model_path
        self.weights = []
        self.runtimes: List[EndpointRuntimeInterface] = []
        self.request_router: DataParallelRequestRouter = DataParallelRequestRouter(
            runtime_selection_policy, total_nodes=len(gpus)
        )
        self.consistent_radix_hash = ConsistentHashingWithRadixCache(
            num_nodes=len(gpus)
        )
        self.gpus = set(gpus)

    # TODO Load runtimes in parallel to reduce cold start time
        # Potentially extract this to the parent model node loder to effeciently load multiple models in parallel
    def load_runtimes(self, model_path, gpus, urls=[], **kwargs):
        def load_runtime(index, gpu):
            runtime: EndpointRuntimeInterface
            if len(urls) > 0:
                runtime = EndpointRuntimeInterface(
                    url=urls[index], 
                    gpu=gpu,
                    **kwargs,
                )
            else:
                runtime = ExtendedSGLangRuntime(
                    model_path=model_path,
                    cuda_devices=[gpu],
                    context_length=1024,
                    mem_fraction_static=0.42,
                    gpu=gpu,
                    **kwargs,
                )
            self.runtimes.append(runtime)
            self.gpus.add(gpu)

        # parallelizae loading for each gpu
        for index, gpu in enumerate(gpus):
            load_runtime(index, gpu)

    def select_runtime_with_identifiers(self, text, sampling_params) -> EndpointRuntimeInterface:
        experiment_id = sampling_params.pop("experiment_id", random_uuid_string())
        request_id = sampling_params.pop("request_id", random_uuid_string())
        runtime_id = self.request_router.select_runtime(text, experiment_id, request_id)
        return self.runtimes[runtime_id]

    def generate_request(self, text, sampling_params):
        runtime: EndpointRuntimeInterface = (
            self.select_runtime_with_identifiers(text, sampling_params)
        )
        start_time = time.time()
        output =  requests.post(
            runtime.generate_url,
            json={
                "text": text,
                "sampling_params": sampling_params,
            },
            timeout=60 * 10,
        ).json()
        output["request_latency"] = time.time() - start_time
        return output
    
    def generate_batch_request(self, batch_kwargs, sampling_params, num_threads):
        with ThreadPoolExecutor(num_threads) as executor:
            futures = []
            for arguments in batch_kwargs:
                futures.append(
                    executor.submit(
                        self.generate_request, arguments, sampling_params
                    )
                )
            rets = [f.result() for f in futures]
            return rets

    def update_runtime_selection_policy(self, runtime_selection_policy, custom_runtime_selector=None):
        self.request_router.update_runtime_selection_policy(runtime_selection_policy)
        self.request_router.custom_selector = custom_runtime_selector

    def clear_kv_cache(self):
        for runtime in self.runtimes:
            requests.get(runtime.flush_cache_url)

    async def async_generate_batch_request_per_sec(
        self,
        requests: Iterable,
        request_rate: float,
        routine,
    ):
        async def get_request(
            input_requests,
            request_rate: float,
        ):
            input_requests = iter(input_requests)
            for request in input_requests:
                yield request
                if request_rate == float("inf"):
                    continue
                interval = np.random.exponential(1.0 / request_rate)
                await asyncio.sleep(interval)

        tasks: List[asyncio.Task] = []
        async for request in get_request(requests, request_rate):
            task = asyncio.create_task(routine(*request))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

    async def async_send_request(
        self, text, sampling_params
    ): 
        start_time = time.time()
        # runtime: EndpointRuntimeInterface = (
        #     self.select_runtime_with_identifiers(text, sampling_params)
        # )
        runtime = await asyncio.to_thread(
            self.select_runtime_with_identifiers, text, sampling_params
        )
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(runtime.generate_url,
                    json={
                        "text": text,
                        "sampling_params": sampling_params,
                    },) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        chunks.append(chunk)
                output = b"".join(chunks).decode("utf-8")
                output = json.loads(output)

                # Re-send the request if it failed.
                if "error" not in output:
                    break
        output["request_latency"] = time.time() - start_time
        # print(f"{id} finishes")
        return output

