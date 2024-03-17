import asyncio
import logging

import uvloop
import zmq
import zmq.asyncio
from sglang.srt.backend_config import GLOBAL_BACKEND_CONFIG
from sglang.srt.managers.router.model_rpc import ModelRpcClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback
from sglang.srt.managers.io_struct import SchedulingMetricsReqInput
import time

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class RouterManager:
    def __init__(self, model_client: ModelRpcClient, port_args: PortArgs):
        # Init communication
        context = zmq.asyncio.Context(2)
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{port_args.detokenizer_port}"
        )

        self.send_to_tokenizer = context.socket(zmq.PUSH)
        self.send_to_tokenizer.connect(f"tcp://127.0.0.1:{port_args.tokenizer_port}")

        # Init status
        self.model_client = model_client
        self.recv_reqs = []

        # Init some configs
        self.extend_dependency_time = GLOBAL_BACKEND_CONFIG.extend_dependency_time

    async def loop_for_forward(self):
        while True:
            next_step_input = list(self.recv_reqs)
            self.recv_reqs = []
            out_pyobjs = await self.model_client.step(next_step_input)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

            # async sleep for receiving the subsequent request and avoiding cache miss
            if len(out_pyobjs) != 0:
                has_finished = any([obj.finished for obj in out_pyobjs])
                if has_finished:
                    await asyncio.sleep(self.extend_dependency_time)

            await asyncio.sleep(0.0006)
    
    async def scheduler_metrics_request(self, recv_req: SchedulingMetricsReqInput):
        """
        Pipes the scheduler request model client to get the metrics back to detokenizer flow.

        Detokenizer used in order to follow structure of existing code.
        """
        start = time.time()
        out = await self.model_client.scheduler_metrics_request(recv_req)
        inner_time = time.time() - start
        out.inner_router_time = inner_time
        self.send_to_tokenizer.send_pyobj(out)

    async def loop_for_recv_requests(self, loop):
        """
        Recieves from tokenizer and forwards to model. 

        In the case that it's a scheduling metric request, it will be handled asynchronously
        """
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            if isinstance(recv_req, SchedulingMetricsReqInput):
                loop.create_task(self.scheduler_metrics_request(recv_req))
                continue
            self.recv_reqs.append(recv_req)


def start_router_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        model_client = ModelRpcClient(server_args, port_args)
        router = RouterManager(model_client, port_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_recv_requests(loop))

    loop.run_until_complete(router.loop_for_forward())
