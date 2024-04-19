from typing import List, Optional, Union, Tuple, Dict
import uuid
import os
import json
from abc import ABC, abstractclassmethod
import heapq
import numpy as np
import time
from transformers import AutoTokenizer
import logging
import torch
import random
from dataclasses import asdict
from enum import Enum

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    generate_chat_conv,
    register_conv_template,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.router.manager import RouterManager
from sglang.srt.managers.router.model_rpc import ModelRpcClient, ModelRpcServer
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.router.model_runner import GPUConfig
from data_parallel_request_cache import DataParallelRequestRouter, DataParallelRuntimeSelectionPolicy
from sglang.srt.managers.io_struct import (
    BatchStrOut,
    BatchTokenIDOut,
    DetokenizeReqInput,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedGenerateReqInput,
    SchedulingMetricsReqInput,
    SchedulingMetricsOut,
    DumpTrace
)
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.backend_config import GLOBAL_BACKEND_CONFIG
from sglang.srt.managers.router.infer_batch import Batch
from benchmarks.benchmark_workload_gen import WorkloadPrefixDataLoader
from benchmarks.benchmark_utils import RequestFuncOutput, BenchmarkMetrics

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

def random_uuid_string():
    return str(uuid.uuid4().hex)

# Use simulated ModelRpcServer to maintain node state
class ServerRuntimeSimulator:
    def __init__(
        self,
        gpu_config: GPUConfig,
        model_path: str,
        profile_mode: bool = False,
        tokenizer_path: Optional[str] = None,
        load_format: str = "auto",
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        mem_fraction_static: float = ServerArgs.mem_fraction_static,
        max_prefill_num_token: int = ServerArgs.max_prefill_num_token,
        context_length: int = ServerArgs.context_length,
        tp_size: int = 1,
        schedule_heuristic: str = "lpm",
        attention_reduce_in_fp32: bool = False,
        random_seed: int = 42,
        log_level: str = "error",
        disable_radix_cache: bool = False,
        enable_flashinfer: bool = False,
        disable_regex_jump_forward: bool = False,
        disable_disk_cache: bool = False,
        api_key: str = "",
        port: Optional[int] = None,
        additional_ports: Optional[Union[List[int], int]] = None,
        cuda_devices: Optional[List[int]] = None,
        freeze: bool = False,
        log_prefix_hit: bool = False,
    ):
        host = "0.0.0.0"
        port, additional_ports = 0, [0] * 100
        if profile_mode:
            load_format = 'dummy'
        server_args = ServerArgs(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            host=host,
            port=port,
            additional_ports=additional_ports,
            load_format=load_format,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            mem_fraction_static=mem_fraction_static,
            max_prefill_num_token=max_prefill_num_token,
            context_length=context_length,
            tp_size=tp_size,
            schedule_heuristic=schedule_heuristic,
            attention_reduce_in_fp32=attention_reduce_in_fp32,
            random_seed=random_seed,
            log_level=log_level,
            cuda_devices=cuda_devices,
            freeze=freeze,
            log_prefix_hit=log_prefix_hit,
            disable_radix_cache=disable_radix_cache,
            enable_flashinfer=enable_flashinfer,
            disable_regex_jump_forward=disable_regex_jump_forward,
            disable_disk_cache=disable_disk_cache,
            api_key=api_key,
        )
        self.server_args = server_args
        self.url = random_uuid_string()
        
        port_args = PortArgs(
            tokenizer_port=server_args.additional_ports[0],
            router_port=server_args.additional_ports[1],
            detokenizer_port=server_args.additional_ports[2],
            nccl_port=server_args.additional_ports[3],
            migrate_port=server_args.additional_ports[4],
            model_rpc_ports=server_args.additional_ports[5:],
        )
        # NOTE: some metadata is maintained in GPU memory, be careful when #replicas is too high
        self.model_rpc = ModelRpcServer(0, server_args, port_args, 
                                        simulate=not profile_mode, gpu_config=gpu_config)
        self.manager_recv_reqs = []
        self.gpu_config = gpu_config
        # # Event in each queue will start from these time stamps
        # self.tokenizer_next_start_time = 0
        # self.manager_next_start_time = 0
        # self.detokenizer_next_start_time = 0
        self.local_clock = 0.0
        self.tokenizer_clock = 0.0
        self.manager_clock = 0.0
    
    def reset_clock(self):
        self.local_clock = 0.0
        self.tokenizer_clock = 0.0
        self.manager_clock = 0.0
    
    def simulate_step(self, recv_reqs) -> int:
        for recv_req in recv_reqs:
            self.model_rpc.handle_generate_request(recv_req)
        forward_times = self.model_rpc.forward_step(self.gpu_config.forward_simulation)
        forward_time = sum(forward_times)
        ret = self.model_rpc.out_pyobjs
        self.model_rpc.out_pyobjs = []
        return forward_time, ret

    class Process(Enum):
        TOKENIZER = 0
        MANAGER = 1
        DETOKENIZER = 2
        
class SimulationEvent(ABC):
    """
    runtime_id is from external world, e.g. request generator
    """
    def __init__(self, task: str, time: float, runtime_id: int):
        self.task = task
        self.time = time
        self.runtime_id = runtime_id
    
    @abstractclassmethod
    def advance_to_schedule_time(self, simulator: "Simulation"):
        pass
    
    # call this before spawning the event
    def update_lock(self, overhead, simulator: "Simulation", thread: ServerRuntimeSimulator.Process=None):
        if self.runtime_id == -1:
            simulator.global_clock += overhead
        else:
            runtime: ServerRuntimeSimulator = simulator.runtimes[self.runtime_id]
            if thread is ServerRuntimeSimulator.Process.TOKENIZER:
                runtime.tokenizer_clock = max(runtime.tokenizer_clock, self.time)
                runtime.tokenizer_clock += overhead
            if thread is ServerRuntimeSimulator.Process.MANAGER:
                runtime.manager_clock = max(runtime.manager_clock, self.time)
                runtime.manager_clock += overhead
            runtime.local_clock = max(runtime.tokenizer_clock, runtime.manager_clock)
            simulator.global_clock = max(simulator.global_clock, runtime.local_clock)
            
    def wrapper_process_event(self, simulator: "Simulation"):
        runtime = simulator.runtimes[self.runtime_id]
        logging.debug(f"{self.runtime_id} processing {self.task} scheduled at {self.time}, global clock: {simulator.global_clock}, local lock: {runtime.local_clock}, {runtime.tokenizer_clock}, {runtime.manager_clock}")
        self.process_event(simulator)
    
    @abstractclassmethod
    def process_event(self, simulator: "Simulation"):
        pass
    
    def __lt__(self, other):
        return self.time < other.time
    
    def __eq__(self, other):
        return self.time == other.time
    
class Simulation:
    def __init__(self, runtimes: List[ServerRuntimeSimulator], router: DataParallelRequestRouter):
        self.global_clock = 0.0
        self.runtimes: List[ServerRuntimeSimulator] = runtimes
        self.router = router
        self.events = []
        # rid -> RequestFuncOutput
        self.request_output: Dict[str, RequestFuncOutput] = {}
        self.unfinished_requests = 0
        self.rid_to_input = {} # rid -> input request
 
    def add_event(self, event: SimulationEvent):
        heapq.heappush(self.events, event)
    
    def reset_state(self):
        self.global_clock = 0.0
        self.events = []
        self.request_output = {}
        for runtime in self.runtimes:
            runtime.reset_clock()
    
    def warm_up(self):
        logging.info('--- Warm up started ---')
        prompt = "Say this is a warmup request."
        warm_up_request = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
            },
            "input_ids": [0] * 9,
        }
        for i in range(len(self.runtimes)):
            self.request_output[str(i)] = RequestFuncOutput(
                prompt_len=len(warm_up_request['input_ids']), 
                send_out_time=0.0,
                route_dest=i,
            )
            generate_input = GenerateReqInput(
                text=prompt,
                sampling_params=warm_up_request['sampling_params'],
                rid=str(i),
                stream=True,
            )
            GenerateRequestEvent(0.0, generate_input, i).process_event(self)
        while True:
            if all(r.success for r in self.request_output.values()):
                break
            for i in range(len(self.runtimes)):
                ModelStepEvent(0.0, i).process_event(self)
        self.reset_state()
        logging.info('--- Warm up finished ---')
        
    def run(self) -> List[RequestFuncOutput]:
        previous_stamp = self.global_clock
        while self.events:
            if self.global_clock > self.time_litmit:
                break
            if not self.unfinished_requests:
                break
            if self.global_clock - previous_stamp >= 10.0:
                logging.info(f"------------ Global clock: {self.global_clock} ------------")
                previous_stamp = self.global_clock
            event: SimulationEvent = heapq.heappop(self.events)
            event.advance_to_schedule_time(self)
            event.wrapper_process_event(self)
        all_req_outputs = [{rid: asdict(rq)} for rid, rq in self.request_output.items()]
        with open("output.json", "w") as f:
            f.write(json.dumps(all_req_outputs, indent=4))
        return [rq for rq in self.request_output.values()]
        # return list(self.request_output.values())
    
    # requests: List[Dict[str, Dict, List]]
    def initialize_all_request_with_rps(
        self, 
        requests, 
        rps,
        time,
        send_out_times=None, 
    ):
        self.time_litmit = time
        if rps != float('inf') and time != float('inf'):
            assert len(requests) >= int(rps * time)
        
        if not send_out_times:
            send_time = self.global_clock
            for request in requests:
                self.add_event(SendRequestEvent(send_time, request))
                if rps == float('inf'):
                    interval = 0
                else:
                    interval = np.random.exponential(1 / rps)
                send_time += interval
        else:
            for request, send_time in zip(requests, send_out_times):
                self.add_event(SendRequestEvent(send_time, request))
        self.unfinished_requests = len(requests)
    
    def start_model_forwarding_loop(self):
        for i in range(len(self.runtimes)):
            self.add_event(ModelStepEvent(0.0, i))

class SendRequestEvent(SimulationEvent):
    """Send Request
    1. Choose the runtime to send the request to
    2. Spwan a generate request event at server side in case want to simulate network latency
    """
    def __init__(self, time: float, request):
        super().__init__("send_request", time, -1)
        self.request = request
        
    def advance_to_schedule_time(self, simulator: "Simulation"):
        simulator.global_clock = max(simulator.global_clock, self.time)
        
    def select_and_prepare_input(self, simulator: Simulation, text, sampling_params, input_ids, rid=None):
        experiment_id = sampling_params.pop("experiment_id", random_uuid_string())
        if rid is None:
            rid = random_uuid_string()
        runtime_id = simulator.router.select_runtime(text, experiment_id, rid, input_ids, sampling_params=sampling_params)
        generate_input = GenerateReqInput(
            text=text,
            sampling_params=sampling_params,
            rid=rid,
            stream=True,
        )
        return runtime_id, generate_input

    def process_event(self, simulator: Simulation):
        start = time.time()
        runtime_id, generate_input = self.select_and_prepare_input(simulator, **self.request)
        runtime = simulator.runtimes[runtime_id]
        overhead = time.time() - start
        if overhead > 0.03:
            logging.debug(f"Select runtime overhead: {overhead}")

        self.update_lock(overhead, simulator)
        simulator.request_output[generate_input.rid] = RequestFuncOutput(
            rid=generate_input.rid,
            prompt_text=self.request['text'][:20],
            prompt_len=len(self.request['input_ids']), 
            send_out_time=self.time,
            route_dest=runtime_id,
            scheduling_overhead=overhead,
            runtime_selected=runtime_id,
            max_new_tokens=self.request['sampling_params']['max_new_tokens'],
        )
        simulator.rid_to_input[generate_input.rid] = self.request
        simulator.add_event(GenerateRequestEvent(self.time, generate_input, runtime_id))

class GenerateRequestEvent(SimulationEvent):
    """Generate Request
    1. Tokenize the input text
    2. Append to manager's waiting queue
    """
    def __init__(self, time: float, request: GenerateReqInput, runtime_id: int):
        super().__init__("generate_request", time, runtime_id)
        self.request = request
        
    def advance_to_schedule_time(self, simulator: Simulation):
        runtime = simulator.runtimes[self.runtime_id]
        runtime.tokenizer_clock = max(runtime.tokenizer_clock, self.time)
        runtime.local_clock = max(runtime.local_clock, self.time)
        
    def process_event(self, simulator: Simulation):
        start = time.time()
        simulator.request_output[self.request.rid].arrival_time = self.time
        obj = self.request
        obj.post_init()
        is_single = isinstance(obj.text, str)
        runtime = simulator.runtimes[self.runtime_id]
        tokenizer = runtime.model_rpc.tokenizer
        if is_single:
            rid = obj.rid
            # Original impl does not multi-thread this, add here to simulate tokenize latency
            input_ids = tokenizer.encode(obj.text)
            sampling_params = SamplingParams(**obj.sampling_params)
            if sampling_params.max_new_tokens != 0:
                sampling_params.normalize(tokenizer)
                sampling_params.verify()

            if isinstance(obj.image_data, list) and len(obj.image_data) > 0:
                raise NotImplementedError("Image data not supported")
            elif isinstance(obj.image_data, str):
                raise NotImplementedError("Image data not supported")
            else:
                pixel_values, image_hash, image_size = None, None, None
            tokenized_obj = TokenizedGenerateReqInput(
                rid=rid,
                input_text=obj.text,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_hash=image_hash,
                image_size=image_size,
                sampling_params=sampling_params,
                return_logprob=obj.return_logprob,
                logprob_start_len=obj.logprob_start_len,
                stream=obj.stream,
                arrival_time=self.time,
            )
            overhead = time.time() - start
            self.update_lock(overhead, simulator, ServerRuntimeSimulator.Process.TOKENIZER)
            # self.update_lock(0, simulator, ServerRuntimeSimulator.Process.TOKENIZER)
            # logging.debug(f"{self.runtime_id}: tokenized req added at {runtime.tokenizer_clock}, overhead {overhead}")
            # runtime.manager_recv_reqs.append(tokenized_obj)
            simulator.add_event(AddToManagerQueueEvent(runtime.tokenizer_clock, tokenized_obj, self.runtime_id))
        else:
            raise NotImplementedError("Batch request not supported")

class AddToManagerQueueEvent(SimulationEvent):
    def __init__(self, time, tokenized_obj, runtime_id):
        super().__init__("add_to_manager_queue", time, runtime_id)
        self.tokenized_obj = tokenized_obj
    
    def advance_to_schedule_time(self, simulator: Simulation):
        pass
    
    def process_event(self, simulator: Simulation):
        runtime = simulator.runtimes[self.runtime_id]
        simulator.request_output[self.tokenized_obj.rid].append_to_queue_time = self.time
        runtime.manager_recv_reqs.append(self.tokenized_obj)


class ModelStepEvent(SimulationEvent):
    def __init__(self, time: float, runtime_id: int):
        super().__init__("model_step", time, runtime_id)
        
    def advance_to_schedule_time(self, simulator: "Simulation"):
        runtime = simulator.runtimes[self.runtime_id]
        runtime.manager_clock = max(runtime.manager_clock, self.time)
        runtime.local_clock = max(runtime.local_clock, self.time)
        
    def update_metric(self, simulator: Simulation, out_pyobjs: List[BatchTokenIDOut]):
        if out_pyobjs:
            runtime = simulator.runtimes[self.runtime_id]
            meta = '\n'.join(str(obj.brief()) for obj in out_pyobjs)
            logging.debug(f"{self.runtime_id} output obj: {meta}")
            for obj in out_pyobjs:
                for rid, output_token_ids, finished in zip(obj.rids, obj.output_tokens, obj.finished):
                    request_output = simulator.request_output[rid]
                    if not request_output.ttft:
                        request_output.ttft = runtime.manager_clock - request_output.send_out_time
                    request_output.request_latency = runtime.manager_clock - request_output.send_out_time
                    request_output.output_len = len(output_token_ids)
                    request_output.tpot = (request_output.request_latency - request_output.ttft) / request_output.output_len
                    if finished:
                        request_output.success = True
                        request_output.global_time = runtime.manager_clock
                        simulator.unfinished_requests -= 1
                        if rid in simulator.rid_to_input:
                            request = simulator.rid_to_input[rid]
                            text = request['text']
                            input_ids = request['input_ids']
                            simulator.router.finish_request(text=text, request_id=rid, input_ids=input_ids, experiment_id=None, func_output=request_output)
    
    def process_event(self, simulator: Simulation):
        start = time.time()
        runtime = simulator.runtimes[self.runtime_id]
        next_step_input = list(runtime.manager_recv_reqs)
        runtime.manager_recv_reqs = []
        forward_time, out_pyobjs = runtime.simulate_step(next_step_input)
        sleep_time = 0.0006
        # sleep_time = 0.01
        if len(out_pyobjs) != 0:
            has_finished = any([obj.finished for obj in out_pyobjs])
            if has_finished:
                if GLOBAL_BACKEND_CONFIG.extend_dependency_time > 0:
                    sleep_time = GLOBAL_BACKEND_CONFIG.extend_dependency_time
        overhead = time.time() - start + forward_time + sleep_time
        self.update_lock(overhead, simulator, ServerRuntimeSimulator.Process.MANAGER)
        self.update_metric(simulator, out_pyobjs)
        # logging.info(f"{self.runtime_id}: new step scheduled at manager time {runtime.manager_clock:.4f}, total {overhead:.4f}, overhead {overhead - forward_time - sleep_time:.4f}, model {forward_time:.4f}")
        simulator.add_event(ModelStepEvent(runtime.manager_clock, self.runtime_id))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    random.seed(2333)
    np.random.seed(2333)
    gpu_configs = [
        GPUConfig(gpu_id=0, url=None, use_ssh=False),
        GPUConfig(gpu_id=1, url=None, use_ssh=False)
    ]
    def forward_simulation(batch: Batch):
        return 1
    for config in gpu_configs:
        config.regist_simulator_config(forward_simulation, 1 << 30)

    model_name = "mistralai/Mistral-7B-v0.1"
    runtimes = [ServerRuntimeSimulator(gpu_config=config, model_path=model_name) for config in gpu_configs]
    vocab_size = runtimes[0].model_rpc.model_config.vocab_size
    
    router = DataParallelRequestRouter(DataParallelRuntimeSelectionPolicy.RANDOM, total_nodes=2)
    simulator = Simulation(runtimes, router)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rps, exp_time = 8, 30
    num_requests = int(rps * exp_time)
    num_workloads = 10
    dataloader = WorkloadPrefixDataLoader(
        num_workloads,
        num_requests,
        tokenizer,
        num_in_context_examples=4,
        output_len=10,
    )
    requests = dataloader.generate_workload(k=1)
    simulator.initialize_all_request_with_rps(requests, 8, 30)
    simulator.start_model_forwarding_loop()

    results = simulator.run()
    bench_metrics = BenchmarkMetrics.gen_benchmark_metrics(
        tokenizer=tokenizer,
        req_func_outputs=results,
        overall_latency=exp_time,
        time_limit=exp_time,
        gpu_counts=len(gpu_configs),
    )
    exp_params = f"{model_name}, {num_workloads}, {0}, {num_requests}, {rps}, {exp_time}"
    bench_metrics.to_log_file(exp_params)
