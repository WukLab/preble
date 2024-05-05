import requests
import zmq.asyncio
from data_parallel_request_cache import (
    DataParallelRuntimeSelectionPolicy,
    CustomPolicyType,
)
from data_parallel_request_cache import DataParallelRequestRouter
import aiohttp
import uuid
from dataclasses import dataclass
from sglang.srt.server import Runtime as SGLangServer
from typing import List, Iterable, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import json
import asyncio
import zmq
import numpy as np
import time
import paramiko
import sys, traceback
from typing import Tuple
from ssh_runtime import SSHRuntimeManager
from vllm_runtime import VLLMRuntimeManager
from dataclasses import field
from sglang.srt.managers.router.model_runner import GPUConfig # FIXME wrong import
from simulator import ServerRuntimeSimulator, Simulation
from benchmarks.benchmark_utils import RequestFuncOutput
from sglang.srt.managers.router.infer_batch import Batch
from sglang.srt.managers.io_struct import BatchStrOut
import torch
import logging
from benchmarks.benchmark_utils import BenchmarkMetrics, MajorExperimentArgs, WorkloadConfig, ExperimentType
from benchmarks.multi_experiment_benchmark_utils import RequestRateManager, Workload

logger = logging.getLogger(__name__)

def random_uuid_string():
    return str(uuid.uuid4().hex)

@dataclass
class EndpointRuntimeInterface:
    hit_ratio: float = 0.0
    
    def __post_init__(self):
        self.runtime_id = str(uuid.uuid4())
        assert self.url is not None
        self._generate_url = f"{self.url}/generate" \
            if getattr(self, "_generate_url", None) is None \
                else self._generate_url
    
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

    def prepare_request_payload(self, 
                                text: str, 
                                sampling_params: dict, 
                                rid: str, 
                                stream: bool):
        return {
            "text": text,
            "sampling_params": sampling_params,
            "rid": rid,
            "stream": stream
        }
    
    def process_stream_output(self,
                              chunk: dict,
                              output: RequestFuncOutput,
                              **kwargs):
        assert 'current_experiment_state_time' in kwargs
        current_experiment_state_time = kwargs['current_experiment_state_time']
        output.generated_text = chunk["text"]
        output.output_len = chunk['meta_info']['completion_tokens']
        output.arrival_time = chunk['meta_info']['arrival_time'] - current_experiment_state_time
        output.append_to_queue_time = chunk['meta_info']['append_to_queue_time'] - current_experiment_state_time

@dataclass
class URLRuntime(EndpointRuntimeInterface):
    def __init__(self, url, gpu):
        self.url = url
        self.gpu = gpu
        self.hit_ratio_url = f"{self.url}/windowed_prefix_hit_ratio"
        super().__init__()

class ExtendedSGLangRuntime(SGLangServer, EndpointRuntimeInterface):
    def __init__(self, gpu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu = gpu

class SSHRuntime(SSHRuntimeManager, EndpointRuntimeInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class VLLMRuntime(VLLMRuntimeManager, EndpointRuntimeInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_request_payload(self, 
                                text: str, 
                                sampling_params: dict, 
                                rid: str, 
                                stream: bool):
        # vllm runtime uses different keys for max tokens
        if "max_new_tokens" in sampling_params:
            sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")
        return {
            "prompt": text,
            "sampling_params": sampling_params,
            "rid": rid,
            "model": self.model_path,
            "stream": stream
        }
    
    def process_stream_output(self,
                              chunk: dict,
                              output: RequestFuncOutput,
                              **kwargs):
        choices = chunk["choices"]
        output.generated_text += choices[0]["text"]
        if chunk.get('usage', None) is not None:
            output.output_len = chunk['usage']['completion_tokens']
        
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

        self.scheduling_overheads = []
        self.num_iters = 0

    def load_runtimes(self, model_path, gpu_configs, **kwargs):
        logger.info(kwargs)
        
        def load_runtime(config: GPUConfig):
            runtime: EndpointRuntimeInterface
            gpu_id = config.gpu_id
            if self.simulate:
                runtime = SimulationRuntime(
                    model_path=model_path,
                    gpu_config=config,
                    **config.runtime_args,
                )
            elif config.use_ssh and config.vllm_config is None:
                runtime = SSHRuntime(
                    model_path=model_path,
                    ssh_config=config.ssh_config,
                    gpu=gpu_id,
                    cuda_devices=gpu_id,
                    **config.runtime_args
                )
            elif config.url:
                runtime = URLRuntime(
                    url=config.url,
                    gpu=gpu_id, 
                    # cuda_devices=[gpu_id],
                    # **config.runtime_args
                    )
            elif config.vllm_config is not None:
                runtime = VLLMRuntime(
                    model_path=model_path,
                    ssh_config=config.ssh_config,
                    gpu=gpu_id,
                    **config.vllm_config,
                    **config.runtime_args,
                )
            else:
                runtime = ExtendedSGLangRuntime(
                    model_path=model_path,
                    cuda_devices=gpu_id,
                    gpu=gpu_id,
                    gpu_config=config,
                    **config.runtime_args,
                )
            #  VLLM Runtime
            self.runtimes.append(runtime)

        # parallelizae loading for each gpu
        for config in gpu_configs:
            load_runtime(config)

    def select_runtime_with_identifiers(self, text, sampling_params, input_ids, *args, **kwargs) -> Tuple[int, str]:
        experiment_id = sampling_params.pop("experiment_id", random_uuid_string())
        request_id = random_uuid_string()
        # For now ignore sampling_params request_id
        runtime_idx = self.request_router.select_runtime(text, experiment_id, request_id, input_ids, sampling_params, *args, **kwargs)
        return runtime_idx, request_id

    def finish_request(self, text, sampling_params, input_ids, func_output: RequestFuncOutput) -> EndpointRuntimeInterface:
        request_id = sampling_params.get("request_id")
        self.request_router.finish_request(text, None, request_id, input_ids, func_output)

    def async_wrap(f):
        async def _func(*args, **kwargs):
            return f(*args, **kwargs)

        return _func
    
    @async_wrap
    def async_select_runtime_with_identifiers(self, text, sampling_params, input_ids) -> Tuple[int, str]:
        return self.select_runtime_with_identifiers(text, sampling_params, input_ids)

    def update_runtime_selection_policy(self, runtime_selection_policy, custom_runtime_selector):
        self.request_router.update_runtime_selection_policy(runtime_selection_policy)
        self.request_router.custom_selector = custom_runtime_selector

    def clear_kv_cache(self):
        for runtime in self.runtimes:
            requests.get(runtime.flush_cache_url)
    
    def get_experiment_results_for_experiment_type(
        self,
        workload_config: Workload,
        experiment_type: ExperimentType
    ):
        if experiment_type == ExperimentType.default: # Should work like the default code
            return self.get_experiment_results(workload_config)
        else:
            raise NotImplementedError("Only DEFAULT experiment type is supported")

    def get_experiment_results(
        self,
        workload_config: Workload,
    ):
        if self.simulate:
            simulator = Simulation(self.runtimes, self.request_router)
            # simulator.warm_up()
            assert len(workload_config.request_groups) == 1, "simulator only supports one group of requests for now"
            assert workload_config.request_groups[0].request_type == ExperimentType.default, "simulator don't support sequential requests yet"
            requests = workload_config.request_groups[0].requests
            request_rate = workload_config.request_groups[0].request_rate
            exp_time = workload_config.exp_time
            send_out_times = workload_config.request_groups[0].send_out_times
            simulator.initialize_all_request_with_rps(requests, request_rate, exp_time, send_out_times)
            simulator.start_model_forwarding_loop()
            return simulator.run()
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Add hit_ratio update background loop
            for i in range(len(self.runtimes)):
                loop.create_task(self.loop_for_hit_ratio_update(i))
                time.sleep(1/len(self.runtimes))
                
            results: List[RequestFuncOutput] = loop.run_until_complete(
                self.async_generate_batch_request_per_sec(
                    workload_config,
                )
            )
        return results
    
    async def update_hit_ratio(self, runtime_id):
        runtime: EndpointRuntimeInterface = self.runtimes[runtime_id]
        async with aiohttp.ClientSession() as session:
            async with session.get(runtime.hit_ratio_url) as response:
                obj = await response.json()
                runtime.hit_ratio = obj["hit_ratio"]
                # logger.info(f'GPU: {runtime_id} hit_ratio = {runtime.hit_ratio}')
    
    async def loop_for_hit_ratio_update(self, runtime_id):
        while True:
            await asyncio.sleep(0.5)
            await self.update_hit_ratio(runtime_id)

    async def get_request(self, input_requests, request_rate: float, send_times: Optional[List[float]] = None):
        input_requests = iter(input_requests)
        for i, request in enumerate(input_requests):
            yield request
            if request_rate == float("inf") and not send_times:
                continue
            if send_times:
                interval = send_times[i + 1] - send_times[i] if i + 1 < len(send_times) else 0
            else:
                interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)

    async def async_generate_batch_request_per_sec(
        self,
        workload_config: Workload,
    ):
        request_manager = RequestRateManager(workload_config.request_groups)
        self.current_experiment_state_time = time.time()
        # Add cache update loop
        loop = asyncio.get_event_loop()
        if workload_config.policy is DataParallelRuntimeSelectionPolicy.CUSTOM and workload_config.custom_policy is CustomPolicyType.GlobalSchedulerTimeWithEviction:
            if workload_config.server_configs[0].runtime_args.get("enable_iterative_eviction", False):
                loop.create_task(self.request_router.custom_selector.cache.update_loop())
        if self.start_time is None:
            self.start_time = time.time()
        tasks: List[asyncio.Task] = []
        try:
            async for workload_id, request in request_manager.get_request():
                task = asyncio.create_task(self.async_send_request(**request))
                # so that it use the current workload_id in the callback
                task.add_done_callback(lambda _, workload_id=workload_id: request_manager.mark_current_req_complete(workload_id))
                tasks.append(task)
            if workload_config.exp_time != float("inf"):
                remaining_time = max(0.5, workload_config.exp_time - (time.time() - self.current_experiment_state_time))
                print(f"Waiting for remaining time", remaining_time)
                done, pending = await asyncio.wait(tasks, timeout=remaining_time)
            else:
                done, pending = await asyncio.wait(tasks)
            for task in pending:
                task.cancel()
            request_manager.cleanup()  # Cancel all running workload loops if not already done
            return [task.result() for task in done]

        except asyncio.CancelledError:
            # Cancel all tasks if a CancelledError occurs
            for task in tasks:
                task.cancel()
            # Wait for all tasks to be cancelled
            await asyncio.gather(*tasks, return_exceptions=True)
            await request_manager.cleanup()  # Cancel all running workload loops if not already done
            # Raise a single CancelledError
            raise
    
    async def async_grouped_batch_request(
        self,
        grouped_requests: Dict[float, Iterable[Dict[str, Any]]],
        exp_time: float = 0.0,
        send_out_times: List[List[float]] = None,
        
    ):
        self.current_experiment_state_time = time.time()

        if self.start_time is None:
            self.start_time = time.time()

        tasks: List[asyncio.Task] = []

        try:
            for request_rate, requests in grouped_requests.items():
                send_times = send_out_times.get(request_rate) if send_out_times else None
                async for request in self.get_request(requests, request_rate, send_times):
                    task = asyncio.create_task(self.async_send_request(**request))
                    tasks.append(task)

            if exp_time != float("inf"):
                remaining_time = max(0.5, exp_time - (time.time() - self.current_experiment_state_time))
                print(f"Waiting for remaining time: {remaining_time}")
                done, pending = await asyncio.wait(tasks, timeout=remaining_time)
            else:
                done, pending = await asyncio.wait(tasks)

            for task in pending:
                task.cancel()
            self.num_iters = self.request_router.custom_selector.cache.num_iters // 2

            return [task.result() for task in done]

        except asyncio.CancelledError:
            # Cancel all tasks if a CancelledError occurs
            for task in tasks:
                task.cancel()
            # Wait for all tasks to be cancelled
            await asyncio.gather(*tasks, return_exceptions=True)
            # Raise a single CancelledError
            raise


    async def send_request_sequentially(self, requests, request_rate: float, routine):
        """Send requests for a single group, sequentially according to the request rate."""
        interval = 0 if request_rate == float('inf') else 1.0 / request_rate

        for request in requests:
            start_time = time.time()
            await routine(**request)  # Execute the request and wait for it to complete.
            elapsed_time = time.time() - start_time

            if elapsed_time < interval:
                await asyncio.sleep(interval - elapsed_time)

    async def async_send_request(
        self, text=None, sampling_params=None, input_ids=None, rid=None,
    ) -> RequestFuncOutput: 
        start_time = time.time()
        st = time.perf_counter()
        
        # self.request_router.custom_selector.cache.aggregate_eviction_updates()
        # end = time.perf_counter()
        # print(end-st)

        # print(end-st)
        hit_rates = [r.hit_ratio for r in self.runtimes]
        highest_idx = int(np.argmax(hit_rates))
        if hit_rates[highest_idx] < 0.7:
            highest_idx = None
        runtime_idx, request_id = await asyncio.to_thread(
            self.select_runtime_with_identifiers, text, sampling_params, input_ids, runtime_id_with_highest_hit_rate=highest_idx, hit_rates=hit_rates
        )
        scheduling_overhead = time.time() - start_time
        self.scheduling_overheads.append(scheduling_overhead)

        api_url = self.runtimes[runtime_idx].generate_url
        # If request in sampling_params pop it
        if "request_id" in sampling_params:
            sampling_params.pop("request_id")
        payload = self.runtimes[runtime_idx].prepare_request_payload(text, sampling_params, request_id, stream=True)
        output = RequestFuncOutput()
        output.rid = rid
        output.prompt_text = text[:20]
        output.prompt_len = len(input_ids)
        output.send_out_time = start_time - self.current_experiment_state_time
        output.runtime_selected = runtime_idx
        output.num_gpus = len(self.runtimes)



        # runtime = await self.async_select_runtime_with_identifiers(text, sampling_params)
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.request_sent_time.append(time.time() - self.start_time)
            ttft = 0
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
                                output.success = True
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
                            output.request_latency = time.perf_counter() - st
                            self.runtimes[runtime_idx].process_stream_output(
                                data, output, current_experiment_state_time=self.current_experiment_state_time)
                            output.global_time = time.time() - self.current_experiment_state_time
                        # print(data["meta_info"])
                    else:
                        # handle error. This is needed because for vllm, if the context is too long, it will not generate
                        output.error = response.reason
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
        #  throughput as token generated per second
        output.scheduling_overhead = scheduling_overhead
        if output.success:
            output.tpot = (output.request_latency - output.ttft)/max(1, output.output_len)

        sampling_params["request_id"] = request_id
        await asyncio.to_thread(
            self.finish_request, text, sampling_params, input_ids, output
        )
        return output


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
