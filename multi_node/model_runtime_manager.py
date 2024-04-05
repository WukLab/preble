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
import paramiko
import sys, traceback
from ssh_runtime import SSHRuntimeManager
from dataclasses import field
from sglang.srt.managers.router.model_runner import GPUConfig
from simulator import ServerRuntimeSimulator, Simulation
from benchmarks.benchmark_utils import RequestFuncOutput
from sglang.srt.managers.router.infer_batch import Batch
import torch
import logging

def random_uuid_string():
    return str(uuid.uuid4().hex)

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
    
class SSHRuntime(SSHRuntimeManager, EndpointRuntimeInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class SimulationRuntime(ServerRuntimeSimulator, EndpointRuntimeInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ModelDetails:
    """
    Supports Data Parallel Model Allocation
    """

    def __init__(
        self, model_path, gpu_configs,
        simulate=False,
        runtime_selection_policy=DataParallelRuntimeSelectionPolicy.RANDOM,
    ) -> None:
        self.model_path = model_path
        self.weights = []
        self.runtimes: List[EndpointRuntimeInterface] = []
        self.request_router: DataParallelRequestRouter = DataParallelRequestRouter(
            runtime_selection_policy, total_nodes=len(gpu_configs)
        )
        # self.gpus = set(gpus)
        self.gpu_configs = gpu_configs
        self.start_time = None
        self.request_sent_time = []
        self.current_experiment_state_time = None
        self.simulate = simulate

    # TODO Load runtimes in parallel to reduce cold start time
        # Potentially extract this to the parent model node loder to effeciently load multiple models in parallel
    # Send context-length like params from input
    def load_runtimes(self, model_path, gpu_configs, **kwargs):
        logging.info(kwargs)
        def load_runtime(config: GPUConfig):
            runtime: EndpointRuntimeInterface
            gpu_id = config.gpu_id
            if self.simulate:
                runtime = SimulationRuntime(
                    model_path=model_path,
                    gpu_config=config,
                    **kwargs,
                )
                # TODO: refactor this mess
                vocab_size = runtime.model_rpc.model_config.vocab_size
                def forward_simulation(batch: Batch):
                    logitis = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
                    new_toks = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
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
                    return logitis, new_toks, forward_time
                config.regist_simulator_config(forward_simulation, config.kv_cache_memory)
            elif config.use_ssh:
                runtime = SSHRuntime(
                    model_path=model_path,
                    ssh_config=config.ssh_config,
                    gpu=gpu_id,
                    cuda_devices=gpu_id,
                    **kwargs
                )
            elif config.url:
                runtime = URLRuntime(
                    config.url, 
                    cuda_devices=[gpu_id],
                    **kwargs)
            else:
                runtime = ExtendedSGLangRuntime(
                    model_path=model_path,
                    cuda_devices=[gpu_id],
                    gpu=gpu_id,
                    **kwargs,
                )
            self.runtimes.append(runtime)

        # parallelizae loading for each gpu
        for config in gpu_configs:
            load_runtime(config)

    def select_runtime_with_identifiers(self, text, sampling_params, input_ids) -> EndpointRuntimeInterface:
        experiment_id = sampling_params.pop("experiment_id", random_uuid_string())
        request_id = sampling_params.pop("request_id", random_uuid_string())
        runtime_id = self.request_router.select_runtime(text, experiment_id, request_id, input_ids)
        return self.runtimes[runtime_id]

    def async_wrap(f):
        async def _func(*args, **kwargs):
            return f(*args, **kwargs)

        return _func
    
    @async_wrap
    def async_select_runtime_with_identifiers(self, text, sampling_params, input_ids) -> EndpointRuntimeInterface:
        return self.select_runtime_with_identifiers(text, sampling_params, input_ids)

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
            
    def get_experiment_results(
        self,
        requests: Iterable,
        request_rate: float,
        exp_time: float = 0.0,
    ):
        if self.simulate:
            simulator = Simulation(self.runtimes, self.request_router)
            simulator.initialize_all_request_with_rps(requests, request_rate, exp_time)
            simulator.start_model_forwarding_loop()
            return simulator.run()
        else:
            results: List[RequestFuncOutput] = asyncio.run(
                self.async_generate_batch_request_per_sec(
                    requests,
                    request_rate,
                    self.async_send_request,
                    exp_time,
                )
            )
        return results

    async def async_generate_batch_request_per_sec(
        self,
        requests: Iterable,
        request_rate: float,
        routine,
        exp_time: float = 0.0,
    ):
        self.current_experiment_state_time = time.time()
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
        if self.start_time is None:
            self.start_time = time.time()
        tasks: List[asyncio.Task] = []
        try:
            async for request in get_request(requests, request_rate):
                task = asyncio.create_task(routine(**request))
                tasks.append(task)

            if exp_time != float("inf"):
                remaining_time = max(0.5, exp_time - (time.time() - self.current_experiment_state_time))
                done, pending = await asyncio.wait(tasks, timeout=remaining_time)
            else:
                done, pending = await asyncio.wait(tasks)
            for task in pending:
                task.cancel()
            return [task.result() for task in done]
        except asyncio.CancelledError:
            # Cancel all tasks if a CancelledError occurs
            for task in tasks:
                task.cancel()
            # Wait for all tasks to be cancelled
            await asyncio.gather(*tasks, return_exceptions=True)
            # Raise a single CancelledError
            raise

    async def async_send_request(
        self, text=None, sampling_params=None, input_ids=None
    ) -> RequestFuncOutput: 
        start_time = time.time()
        rid = random_uuid_string()
        sampling_params["request_id"] = rid
        runtime = await asyncio.to_thread(
            self.select_runtime_with_identifiers, text, sampling_params, input_ids
        )
        api_url = runtime.generate_url
        payload = {
            "text": text,
            "sampling_params": sampling_params,
            "rid": rid,
            'stream': True
        }
        output = RequestFuncOutput()
        output.prompt_len = len(input_ids)
        output.send_out_time = start_time - self.current_experiment_state_time

        # runtime = await self.async_select_runtime_with_identifiers(text, sampling_params)
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.request_sent_time.append(time.time() - self.start_time)
            ttft = 0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for chunk in response.content:
                            chunk = chunk.strip()
                            if not chunk:
                                continue

                            chunk = remove_prefix(chunk.decode("utf-8"), "data:").strip()
                            if chunk == "[DONE]":
                                request_latency = time.perf_counter() - st
                            else:
                                data = json.loads(chunk)
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                    most_recent_timestamp)

                                most_recent_timestamp = timestamp

                        output.request_latency = request_latency
                        output.global_time = time.time() - self.current_experiment_state_time
                        output.success = True
                        output.generated_text = data["text"]
                        # print(data["meta_info"])
                    else:
                        output.error = response.reason
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
        #  throughput as token generated per second
        return output

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
