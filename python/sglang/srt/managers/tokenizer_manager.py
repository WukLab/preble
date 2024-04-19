import asyncio
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
from typing import List

import numpy as np
import transformers
import uvloop
import zmq
import zmq.asyncio
from sglang.srt.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_processor,
    get_tokenizer,
)
from sglang.srt.managers.io_struct import (
    BatchStrOut,
    DetokenizeReqInput,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedGenerateReqInput,
    SchedulingMetricsReqInput,
    SchedulingMetricsOut,
    DumpTrace
)
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback, is_multimodal_model, load_image
import uuid
from typing import Dict
import time

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@dataclasses.dataclass
class ReqState:
    out_list: List
    finished: bool
    event: asyncio.Event


global global_processor


def init_global_processor(server_args: ServerArgs):
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )


def get_pixel_values(
    image_data, image_aspect_ratio=None, image_grid_pinpoints=None, processor=None
):
    try:
        processor = processor or global_processor
        image = load_image(image_data)
        image_hash = hash(image_data)
        if image_aspect_ratio == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_processor.image_mean)
            )
            pixel_values = processor.image_processor(image)["pixel_values"][0]
        elif image_aspect_ratio == "anyres":
            pixel_values = process_anyres_image(
                image, processor.image_processor, image_grid_pinpoints
            )
        else:
            pixel_values = processor.image_processor(image)["pixel_values"][0]
        pixel_values = pixel_values.astype(np.float16)
        return pixel_values, image_hash, image.size
    except Exception:
        print("Exception in TokenizerManager:\n" + get_exception_traceback())


class TokenizerManager:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        self.server_args = server_args

        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = context.socket(zmq.PULL)
        self.recv_from_detokenizer.bind(f"tcp://127.0.0.1:{port_args.tokenizer_port}")

        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{port_args.router_port}")

        self.model_path = server_args.model_path
        self.hf_config = get_config(
            self.model_path, trust_remote_code=server_args.trust_remote_code
        )

        # Create a two way zmq pair
        self.context_len = get_context_length(self.hf_config)

        if is_multimodal_model(self.model_path):
            self.processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
            self.tokenizer = self.processor.tokenizer
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.executor = concurrent.futures.ProcessPoolExecutor(
                initializer=init_global_processor,
                mp_context=mp.get_context("fork"),
                initargs=(server_args,),
            )
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )

        self.to_create_loop = True
        self.rid_to_state: Dict[str, ReqState] = {}  # Dict[str -> ReqState]
        self.tokenizer_pool = concurrent.futures.ThreadPoolExecutor()

    async def get_pixel_values(self, image_data):
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints if aspect_ratio == "anyres" else None
        )
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                get_pixel_values,
                image_data,
                aspect_ratio,
                grid_pinpoints,
            )
        else:
            return get_pixel_values(
                image_data, aspect_ratio, grid_pinpoints, self.processor
            )
    
    async def schedule_migration_request(self, url: str):
        self.send_to_router.send_pyobj(url)

    async def get_scheduling_metrics(self, text: str):
        """
        Assume the input isn't tokenized and sends the request to the router to get the individual request metrics
        """
        start_time = time.time()
        # input_ids = self.tokenizer.encode(text)
        loop = asyncio.get_running_loop()
        input_ids = await loop.run_in_executor(self.tokenizer_pool, self.tokenizer.encode, text)
        tokenization_time = time.time() 

        rid = str(uuid.uuid4())
        scheduling_metric_request = SchedulingMetricsReqInput(
            rid=rid,
            input_ids=input_ids,
            tokenizer_dispatch_time=time.time(),
            manager_recv_time=0,
        )
        await self.send_to_router.send_pyobj(scheduling_metric_request)
        routing_time = scheduling_metric_request.tokenizer_dispatch_time
        
        lock = asyncio.Lock()
        event = asyncio.Event()
        state = ReqState([], False, event, lock)
        self.rid_to_state[rid] = state
        await event.wait()
        result = state.out_list[-1]
        del self.rid_to_state[rid]
        event.clear()

        result["tokenization_time"] = tokenization_time - start_time
        result["routing_time"] = routing_time - tokenization_time
        result["return_time"] = time.time()
        result["waiting_time"] = result['return_time'] - routing_time
        return result
    
    async def add_request_to_queue(self, obj: GenerateReqInput):
        if self.to_create_loop:
            await self.create_handle_loop()
        rid = obj.rid
        input_ids = self.tokenizer.encode(obj.text)
        sampling_params = SamplingParams(**obj.sampling_params)
        if sampling_params.max_new_tokens != 0:
            sampling_params.normalize(self.tokenizer)
            sampling_params.verify()

        if isinstance(obj.image_data, list) and len(obj.image_data) > 0:
            pixel_values, image_hash, image_size = await self.get_pixel_values(
                obj.image_data[0]
            )
        elif isinstance(obj.image_data, str):
            pixel_values, image_hash, image_size = await self.get_pixel_values(
                obj.image_data
            )
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
        )
        await self.send_to_router.send_pyobj(tokenized_obj)
    
    async def dump_prefix_hit_trace(self, fpath: str):
        await self.send_to_router.send_pyobj(DumpTrace(fpath))

    async def generate_request(self, obj: GenerateReqInput):
        if self.to_create_loop:
            await self.create_handle_loop()

        arrival_time = time.time()
        is_single = isinstance(obj.text, str)

        if is_single:
            rid = obj.rid
            input_ids = self.tokenizer.encode(obj.text)
            sampling_params = SamplingParams(**obj.sampling_params)
            if sampling_params.max_new_tokens != 0:
                sampling_params.normalize(self.tokenizer)
                sampling_params.verify()

            if isinstance(obj.image_data, list) and len(obj.image_data) > 0:
                pixel_values, image_hash, image_size = await self.get_pixel_values(
                    obj.image_data[0]
                )
            elif isinstance(obj.image_data, str):
                pixel_values, image_hash, image_size = await self.get_pixel_values(
                    obj.image_data
                )
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
                arrival_time=arrival_time,
            )
            self.send_to_router.send_pyobj(tokenized_obj)

            event = asyncio.Event()
            state = ReqState([], False, event)
            self.rid_to_state[rid] = state

            while True:
                await event.wait()
                yield state.out_list[-1]
                state.out_list = []
                if state.finished:
                    del self.rid_to_state[rid]
                    break
                event.clear()
        else:
            assert obj.stream is False
            bs = len(obj.text)
            for i in range(bs):
                rid = obj.rid[i]
                input_ids = self.tokenizer.encode(obj.text[i])
                sampling_params = SamplingParams(**obj.sampling_params[i])
                if sampling_params.max_new_tokens != 0:
                    sampling_params.normalize(self.tokenizer)
                    sampling_params.verify()
                if obj.image_data[i] is None:
                    pixel_values, image_hash, image_size = None, None, None
                else:
                    pixel_values, image_hash, image_size = await self.get_pixel_values(
                        obj.image_data[i]
                    )
                tokenized_obj = TokenizedGenerateReqInput(
                    rid=rid,
                    input_text=obj.text[i],
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_hash=image_hash,
                    image_size=image_size,
                    sampling_params=sampling_params,
                    return_logprob=obj.return_logprob[i],
                    logprob_start_len=obj.logprob_start_len[i],
                    stream=obj.stream,
                    arrival_time=arrival_time,
                )
                self.send_to_router.send_pyobj(tokenized_obj)

                event = asyncio.Event()
                state = ReqState([], False, event)
                self.rid_to_state[rid] = state

            output_list = []
            for i in range(bs):
                rid = obj.rid[i]
                state = self.rid_to_state[rid]
                await state.event.wait()
                output_list.append(state.out_list[-1])
                assert state.finished
                del self.rid_to_state[rid]

            yield output_list

    async def detokenize(self, obj: DetokenizeReqInput):
        token_texts = self.tokenizer.convert_ids_to_tokens(obj.input_ids)
        return [t.decode() if isinstance(t, bytes) else t for t in token_texts]

    async def flush_cache(self):
        flush_cache_req = FlushCacheReq()
        self.send_to_router.send_pyobj(flush_cache_req)

    async def create_handle_loop(self):
        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        loop.create_task(self.handle_loop())

    async def handle_loop(self):
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()

            if isinstance(recv_obj, BatchStrOut):
                for i, rid in enumerate(recv_obj.rids):
                    recv_obj.meta_info[i]["id"] = rid
                    out_dict = {
                        "text": recv_obj.output_str[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                    state = self.rid_to_state[rid]
                    state.out_list.append(out_dict)
                    state.finished = recv_obj.finished[i]
                    state.event.set()
            elif isinstance(recv_obj, SchedulingMetricsOut):
                out_dict = {
                    "waiting_queue_len": recv_obj.waiting_queue_len,
                    "running_req_len": recv_obj.running_req_len,
                    "prefix_match_len": recv_obj.prefix_match_len,
                    "token_kv_available_size": recv_obj.token_kv_available_size,
                    "evicatable_size": recv_obj.evicatable_size,
                    "tree_cache_metrics_hit": recv_obj.tree_cache_metrics_hit,
                    "tree_cache_metrics_total": recv_obj.tree_cache_metrics_total,
                    "input_len": recv_obj.input_len,
                    "total_radix_cache_processing_time": recv_obj.total_radix_cache_processing_time,
                    "queue_processing_time": time.time() - recv_obj.queue_processing_time,
                    'tokenizer_manager_waiting_time': recv_obj.waiting_time_tokenizer_manager,
                    "inner_router_time": recv_obj.inner_router_time,
                    "manager_tokenizer_waiting_time": time.time() - recv_obj.manager_dispatch_time,
                    "manager_recv_time": recv_obj.manager_recv_time,
                    "matching_overhead": recv_obj.matching_overhead,
                }
                state = self.rid_to_state[recv_obj.rid]
                state.out_list.append(out_dict)
                state.finished = True
                state.event.set()
            else:
                raise ValueError(f"Invalid object: {recv_obj}")
