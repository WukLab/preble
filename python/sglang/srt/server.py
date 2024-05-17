"""SRT: SGLang Runtime"""

import asyncio
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from typing import List, Optional, Union

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import aiohttp
import psutil
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.constrained import disable_cache
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.router.manager import start_router_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api_adapter import (
    load_chat_template_for_openai_api,
    v1_chat_completions,
    v1_completions,
)
from sglang.srt.server_args import PortArgs, ServerArgs
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from sglang.srt.managers.router.model_runner import GPUConfig
from sglang.srt.utils import (
    API_KEY_HEADER_NAME,
    APIKeyValidatorMiddleware,
    allocate_init_ports,
    assert_pkg_version,
    enable_show_time_cost,
    get_exception_traceback,
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = logging.getLogger('server')


app = FastAPI()
tokenizer_manager = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/get_model_info")
async def get_model_info():
    result = {
        "model_path": tokenizer_manager.model_path,
    }
    return result


@app.get("/get_server_args")
async def get_server_args():
    return dataclasses.asdict(tokenizer_manager.server_args)


@app.get("/flush_cache")
async def flush_cache():
    await tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


@app.post("/generate")
async def generate_request(obj: GenerateReqInput):
    obj.post_init()
    logger.debug(f"{obj.text[:20]} ...")
    if obj.stream:

        async def stream_results():
            async for out in tokenizer_manager.generate_request(obj):
                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")

    try:
        ret = await tokenizer_manager.generate_request(obj).__anext__()
        return ret
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# Just add request wihout expecting result
@app.post("/add_request")
async def add_request(obj: GenerateReqInput):
    obj.post_init()
    await tokenizer_manager.add_request_to_queue(obj)
    return Response(status_code=200)

@app.post("/migrate_control")
async def migrate_request(migration_target_url: str):
    await tokenizer_manager.schedule_migration_request(migration_target_url)
    return Response(status_code=200)

@app.post("/scheduling_metrics")
async def scheduling_metrics(raw_request: Request):
    """
    Returns metrics that could be used by a global data parallel scheduler.
    The output format is:
    out_dict = {
        "waiting_queue_len": int,
        "running_req_len": int,
        "prefix_match_len": int,
        "token_kv_available_size": int,
        "evicatable_size": int,
        "tree_cache_metrics_hit": int,
        "tree_cache_metrics_total": int,
        "input_len": int
    }
    """
    start_time = time.time()
    request_json = await raw_request.json()
    request = request_json
    if not tokenizer_manager:
        return {
            "status": "error",
            "message": "Tokenizer manager not initialized"
        }
    text = request.get("prompt", None)
    if text is None:
        return {
            "status": "error",
            "message": "Prompt not found in request"
        }
    request_processing_time = time.time() - start_time
    ret = await tokenizer_manager.get_scheduling_metrics(text)
    ret["request_processing_time"] = request_processing_time
    ret["return_time"] = time.time() - ret["return_time"]
    ret["total_internal_request_time"] = time.time() - start_time
    return ret

@app.post("/dump_prefix_hit_trace")
async def dump_prefix_hit_trace(fpath: str):
    """
    Ask the runtime to log prefix hit trace to the provided file path    
    """
    await tokenizer_manager.dump_prefix_hit_trace(fpath)
    return Response(status_code=200)

# {
#     'windowed': recv_obj.windowed,
#     'hit_ratio': recv_obj.hit_ratio,
# }
@app.get('/windowed_prefix_hit_ratio')
async def windowed_prefix_hit_ratio():
    return await tokenizer_manager.handle_windowed_prefix_hit_ratio()

@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    return await v1_completions(tokenizer_manager, raw_request)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    return await v1_chat_completions(tokenizer_manager, raw_request)


    # Non-streaming response.
    ret = await generate_request(adapted_request)
    prompt_tokens = ret["meta_info"]["prompt_tokens"]
    completion_tokens = ret["meta_info"]["completion_tokens"]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=ret["text"]),
        finish_reason=None,  # TODO(comaniac): Add finish reason.
    )
    response = ChatCompletionResponse(
        id=ret["meta_info"]["id"],
        model=request.model,
        choices=[choice_data],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


def launch_server(server_args: ServerArgs, pipe_finish_writer, gpu_config, model_overide_args=None):
    global tokenizer_manager
    global chat_template_name
    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO').upper()
    )

    if server_args.cuda_devices:
       os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in server_args.cuda_devices)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if server_args.show_time_cost:
        enable_show_time_cost()
    if server_args.disable_disk_cache:
        disable_cache()
    if server_args.enable_flashinfer:
        assert_pkg_version("flashinfer", "0.0.4")
    if server_args.chat_template:
        # TODO: replace this with huggingface transformers template
        load_chat_template_for_openai_api(server_args.chat_template)

    # Allocate ports
    server_args.port, server_args.additional_ports = allocate_init_ports(
        server_args.port, server_args.additional_ports, server_args.tp_size
    )
    port_args = PortArgs(
        tokenizer_port=server_args.additional_ports[0],
        router_port=server_args.additional_ports[1],
        detokenizer_port=server_args.additional_ports[2],
        nccl_port=server_args.additional_ports[3],
        migrate_port=server_args.additional_ports[4],
        model_rpc_ports=server_args.additional_ports[5:],
    )
    logger.info(f'{server_args.url()}, ports: {port_args}')

    # Launch processes
    tokenizer_manager = TokenizerManager(server_args, port_args, model_overide_args)
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

    proc_router = mp.Process(
        target=start_router_process,
        args=(
            server_args,
            port_args,
            pipe_router_writer,
            gpu_config,
            model_overide_args,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenizer_process,
        args=(
            server_args,
            port_args,
            pipe_detoken_writer,
        ),
    )
    proc_detoken.start()

    # Wait for the model to finish loading
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print(
            f"Initialization failed. router_init_state: {router_init_state}", flush=True
        )
        print(
            f"Initialization failed. detoken_init_state: {detoken_init_state}",
            flush=True,
        )
        sys.exit(1)
    assert proc_router.is_alive() and proc_detoken.is_alive()

    if server_args.api_key and server_args.api_key != "":
        app.add_middleware(APIKeyValidatorMiddleware, api_key=server_args.api_key)

    print(f"Server is on port {server_args.port} on host {server_args.host} on pid {os.getpid()}")
    def _wait_and_warmup():
        headers = {}
        url = server_args.url()
        if server_args.api_key:
            headers[API_KEY_HEADER_NAME] = server_args.api_key

        # Wait until the server is launched
        for _ in range(120):
            time.sleep(0.5)
            try:
                requests.get(url + "/get_model_info", timeout=5, headers=headers)
                success = True  # Set flag to True if request succeeds
                break
            except requests.exceptions.RequestException as e:
                pass

        # Send a warmup request
        try:
            res = requests.post(
                url + "/generate",
                json={
                    "text": "Say this is a warmup request.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                },
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200
        except Exception as e:
            if pipe_finish_writer is not None:
                pipe_finish_writer.send(get_exception_traceback())
            print(f"Initialization failed. warmup error: {e}")
            raise e

        if pipe_finish_writer is not None:
            pipe_finish_writer.send("init ok")
            
    t = threading.Thread(target=_wait_and_warmup)
    t.start()
    try:
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        t.join()


class Runtime:
    def __init__(
        self,
        log_evel: str = "error",
        gpu_config: Optional[GPUConfig] = None,
        model_overide_args: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_evel, **kwargs)

        # Pre-allocate ports
        self.server_args.port, self.server_args.additional_ports = allocate_init_ports(
            self.server_args.port,
            self.server_args.additional_ports,
            self.server_args.tp_size,
        )

        self.url = self.server_args.url()
        self.generate_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/generate"
        )
        self.hit_ratio_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/windowed_prefix_hit_ratio" 
        )

        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, pipe_writer, gpu_config, model_overide_args),
        )
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        if self.pid is not None:
            try:
                parent = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            psutil.wait_procs(children, timeout=5)
            parent.kill()
            parent.wait(timeout=5)
            self.pid = None

    def get_tokenizer(self):
        return get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
        )

    async def add_request(
        self,
        prompt: str,
        sampling_params,
    ):
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "stream": True,
        }
        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        cur = data["text"][pos:]
                        if cur:
                            yield cur
                        pos += len(cur)
    
    async def add_request_await(
        self,
        prompt: str,
        sampling_params,
    ) -> None:
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                return await response.json()

    def __del__(self):
        self.shutdown()
