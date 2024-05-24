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
import uuid

random.seed(10)
np.random.seed(10)

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data_parallel_request_cache import DataParallelRequestRouter, CustomPolicyType, DataParallelRuntimeSelectionPolicy
from model_runtime_manager import remove_prefix
from benchmarks.benchmark_utils import RequestFuncOutput

logger = logging.getLogger("fastapi")
logging.basicConfig(level=logging.DEBUG)

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
    input_ids: List[int]
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
        # "input_ids": obj.input_ids,
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
        await finished_requests_queue.put(output)

    return StreamingResponse(get_requests(), media_type="text/event-stream")

async def process_req(request: Request):
    try:
        obj = await request.json()
        generate_req_input = GenerateReqInput(**obj)
        return await generate_request_helper(generate_req_input)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

async def process_runtime_selection():
    while True:
        obj, request_id = await runtime_request_queue.get()
        text, input_ids, sampling_params = obj.text, obj.input_ids, obj.sampling_params
        sampling_params = sampling_params.dict()
        # hit_rates = [r.hit_ratio for r in runtimes] # 
        hit_rates = [0 for _ in runtimes] # TODO handle hitrates
        highest_idx = int(np.argmax(hit_rates))
        if (highest_idx is not None) and (hit_rates[highest_idx] < 0.7):
            highest_idx = None
        
        try:
            runtime_id = request_router.select_runtime(text, sampling_params, input_ids, runtime_id_with_highest_hit_rate=highest_idx, hit_rates=hit_rates)
            runtime_events[request_id] = (runtime_events[request_id][0], runtime_id)
        except Exception as e:
            logger.error(f"Error selecting runtime: {e}")
            runtime_events[request_id] = (runtime_events[request_id][0], None)
        finally:
            runtime_events[request_id][0].set()
            runtime_request_queue.task_done()

async def process_cleanup_selection():
    while True:
        output_obj = await finished_requests_queue.get()
        request_router.finish_request(output_obj, experiment_id="exp_id", request_id="rid")

app = FastAPI()

@app.post("/process")
async def process(request: Request):
    return await process_req(request)

if __name__ == "__main__":
    import uvicorn
    from uvicorn import Config, Server
    # Define your runtime selection policy here
    runtime_selection_policy = DataParallelRuntimeSelectionPolicy.ROUND_ROBIN # Placeholder for the actual policy

    runtime_events = {}
    runtime_request_queue = asyncio.Queue()
    finished_requests_queue = asyncio.Queue()

    gpu_config_count = 1  # Number of GPU configs

    # Example runtime nodes
    runtimes = ["http://127.0.0.1:30000/generate"]
    num_nodes = gpu_config_count

    request_router = DataParallelRequestRouter(
        runtime_selection_policy, total_nodes=gpu_config_count
    )

    async def main():
        loop.create_task(process_runtime_selection())
        loop.create_task(process_cleanup_selection())
        config = uvicorn.Config(app=app, loop="asyncio", host="127.0.0.1", port=8000)
        server = uvicorn.Server(config)
        await server.serve()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    # uvicorn.run(app, host="0.0.0.0", loop='asyncio', port=8000)