from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import json
import logging
import asyncio
import time
import numpy as np
import traceback
import sys
import aiohttp
import random
import os
import fire
import uuid
from transformers import AutoTokenizer
from typing import List, Optional
import uvicorn
from sglang.srt.managers.router.model_runner import GPUConfig

random.seed(10)
np.random.seed(10)

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data_parallel_request_cache import DataParallelRequestRouter, CustomPolicyType, DataParallelRuntimeSelectionPolicy

from model_runtime_manager import remove_prefix
from benchmarks.benchmark_utils import RequestFuncOutput
from global_scheduler_with_time import GlobalSchedulerWithTime
from multi_node_loader import MultiNodeLoader

logger = logging.getLogger(__name__)

class SamplingParams(BaseModel):
    max_new_tokens: int = 16
    stop: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    dtype: Optional[str] = None
    regex: Optional[str] = None

class GenerateReqInput(BaseModel):
    text: str
    input_ids: Optional[List[int]]
    sampling_params: SamplingParams
    stream: bool = True

def process_stream_output(chunk: dict, output: RequestFuncOutput, **kwargs):
    current_experiment_state_time = kwargs['current_experiment_state_time']
    output.generated_text += chunk["text"]
    output.output_len = chunk['meta_info']['completion_tokens']
    output.arrival_time = chunk['meta_info']['arrival_time'] - current_experiment_state_time
    output.append_to_queue_time = chunk['meta_info']['append_to_queue_time'] - current_experiment_state_time

async def async_send_request(
    text=None, input_ids=None, payload=None, runtime_id=None, runtime_url=None, rid=None
):
    start_time = time.time()
    st = time.perf_counter()
    scheduling_overhead = time.time() - start_time
    api_url = runtime_url

    output = RequestFuncOutput()
    output.rid = rid
    output.prompt_text = text[:20]
    output.prompt_len = len(input_ids)
    output.runtime_selected = runtime_id
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        ttft = 0
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        yield chunk
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data:").strip()
                        if chunk == "[DONE]":
                            output.success = True
                            break
                        else:
                            data = json.loads(chunk)
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            process_stream_output(data, output, current_experiment_state_time=st)
                        output.request_latency = time.perf_counter() - st
                else:
                    output.error = response.reason
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    #  throughput as token generated per second
    output.scheduling_overhead = scheduling_overhead
    if output.success:
        output.tpot = (output.request_latency - output.ttft) / max(1, output.output_len)
    yield output

async def generate_request_helper(obj: GenerateReqInput):
    request_id = str(uuid.uuid4())
    runtime_events[request_id] = (asyncio.Event(), None)
    await runtime_request_queue.put((obj, request_id))
    await runtime_events[request_id][0].wait()

    runtime_id = runtime_events[request_id][1]
    runtime_events.pop(request_id)

    if runtime_id is None:
        raise HTTPException(status_code=500, detail="Runtime selection failed")
    
    url = runtimes[runtime_id]
    rid = str(uuid.uuid4())
    payload = {
        "text": obj.text,
        # "input_ids": obj.input_ids, TODO some systems support directly sending input ids
        "sampling_params": obj.sampling_params.dict(),
        "rid": rid,
        "stream": True
    }
    text = obj.text
    input_ids = obj.input_ids
    async def get_requests():
        async for chunk in async_send_request(text, input_ids, payload, runtime_id, url, rid):
            if isinstance(chunk, RequestFuncOutput):
                break
            yield chunk
        output = chunk
        await finished_requests_queue.put((output, text, input_ids))

    return StreamingResponse(get_requests(), media_type="text/event-stream")

async def process_req(request: Request):
    try:
        obj = await request.json()
        generate_req_input = GenerateReqInput(**obj)
        # if text doesn't have tokenization/tokenize the input here
        if not generate_req_input.input_ids:
            generate_req_input.input_ids = tokenizer.encode(generate_req_input.text)
        return await generate_request_helper(generate_req_input)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

async def process_runtime_selection():
    while True:
        obj: GenerateReqInput
        obj, request_id = await runtime_request_queue.get()
        text, input_ids, sampling_params = obj.text, obj.input_ids, obj.sampling_params
        sampling_params = sampling_params.dict()
        # hit_rates = [r.hit_ratio for r in runtimes] # 
        # hit_rates = [0 for _ in runtimes] # TODO handle hitrates
        # highest_idx = int(np.argmax(hit_rates))
        # if (highest_idx is not None) and (hit_rates[highest_idx] < 0.7):
        #     highest_idx = None
        highest_idx = None
        hit_rates = [0 for _ in runtimes] # TODO add hot/cold support
        try:
            runtime_id = request_router.select_runtime(text=text, experiment_id="1", input_ids=input_ids, request_id=request_id, sampling_params=sampling_params, runtime_id_with_highest_hit_rate=highest_idx, hit_rates=hit_rates)
            runtime_events[request_id] = (runtime_events[request_id][0], runtime_id)
        except Exception as e:
            logger.error(f"Error selecting runtime: {e}")
            runtime_events[request_id] = (runtime_events[request_id][0], None)
        finally:
            runtime_events[request_id][0].set()
            runtime_request_queue.task_done()

async def process_cleanup_selection():
    while True:
        output_obj, text, input_ids = await finished_requests_queue.get()
        request_router.finish_request(text=text, input_ids=input_ids, func_output=output_obj, experiment_id="exp_id", request_id="rid")

app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    return await process_req(request)


def start_server(runtime_selection_policy="custom", runtime_urls="http://127.0.0.1:30000/generate", host='127.0.0.1', port=8000, model="mistralai/Mistral-7B-v0.1"):
    """
    Starts the server with the specified runtime selection policy, runtime URLs, and model.

    Args:
        runtime_selection_policy (str): The policy for selecting runtimes. Can be "round_robin", "lor", or "custom". Defaults to "custom".
        runtime_urls (str): A comma-separated list of runtime URLs. Defaults to "http://127.0.0.1:30000/generate".
        host (str): The host address for the server. Defaults to '127.0.0.1'.
        port (int): The port number for the server. Defaults to 8000.
        model (str): The model name or path to be used by the server. Defaults to "mistralai/Mistral-7B-v0.1".

    Example:
        preble start_server --runtime_urls="http://127.0.0.1:30000/generate,"http://127.0.0.1:30001/generate" 
    Raises:
        ValueError: If the runtime selection policy is not valid.
    """
    global request_router
    global tokenizer
    global runtimes
    # TODO check that these urls are valid

    tokenizer = AutoTokenizer.from_pretrained(model)

    runtimes = runtime_urls.split(',')
    num_nodes = len(runtimes)
    
    if runtime_selection_policy == "round_robin":
        runtime_selection_policy = DataParallelRuntimeSelectionPolicy.ROUND_ROBIN
    elif runtime_selection_policy == "lor":
        runtime_selection_policy = DataParallelRuntimeSelectionPolicy.LEAST_OUTSTANDING_REQUESTS
    elif runtime_selection_policy == "custom":
        runtime_selection_policy = DataParallelRuntimeSelectionPolicy.CUSTOM
    else:
        raise ValueError("Invalid runtime selection policy")

    global_scheduler = GlobalSchedulerWithTime(num_nodes=num_nodes, enable_eviction=True)
    request_router = DataParallelRequestRouter(
        runtime_selection_policy, total_nodes=num_nodes
    )
    request_router.custom_selector = global_scheduler
    async def main():
        loop.create_task(process_runtime_selection())
        loop.create_task(process_cleanup_selection())
        config = uvicorn.Config(app=app, loop="asyncio", host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

def start_server_and_load_models(model_name="mistralai/Mistral-7B-v0.1", devices=[0, 1], host="127.0.0.1", port=8000):
    """
    Loads the specified model onto the given devices and starts the server.

    Args:
        model_name (str): The name or path of the model to be loaded. Defaults to "mistralai/Mistral-7B-v0.1".
        devices (list): A list of GPU device IDs to load the model onto. Defaults to [0, 1].
        host (str): The host address for the server. Defaults to '127.0.0.1'.
        port (int): The port number for the server. Defaults to 8000.
    
    Example: preble deploy_and_run
    
    Raises:
        KeyboardInterrupt: If the server is interrupted, it unloads the model.
    """
    server_args = {
        'log_prefix_hit': True,
        'mem_fraction_static': 0.8,
        'context_length': 32768,
        "enable_flashinfer": True,
        'schedule_heuristic': 'fcfs-mpq',
        "chunk_prefill_budget": 512,
        'report_hit_ratio': True ,
        'enable_iterative_eviction': True,
    }
    # GPU Configuration
    gpu_configs = [
        GPUConfig(gpu_id=device, url=None, use_ssh=False, runtime_args=server_args)
        for device in devices
    ]

    loader = MultiNodeLoader()
    model_details = loader.load_model(
        model_path=model_name,
        gpu_configs=gpu_configs,
    )
    runtimes = []
    for runtime in model_details.runtimes:
        runtimes.append(runtime.generate_url)
    print(f"Loading runtimes at {runtimes}")
    try:
        start_server(runtime_selection_policy="custom", runtime_urls=",".join(runtimes), model=model_name, host=host, port=port)
    except KeyboardInterrupt:
        print("Unloading model")
        loader.unload_model(model_details)

runtime_events = {}
runtime_request_queue = asyncio.Queue()
finished_requests_queue = asyncio.Queue()
request_router = None

def main():
    fire.Fire({
        "run": start_server,
        "deploy_and_run": start_server_and_load_models
    })

if __name__ == "__main__":
    main()
