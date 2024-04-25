import asyncio
import logging
from typing import Dict
import uuid

import uvloop
import zmq
import zmq.asyncio
from sglang.srt.backend_config import GLOBAL_BACKEND_CONFIG
from sglang.srt.managers.router.model_rpc import ModelRpcClient
from sglang.srt.managers.tokenizer_manager import ReqState
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback
from sglang.srt.managers.io_struct import SchedulingMetricsReqInput, MigrationReq, DumpTrace
from sglang.srt.managers.router.model_runner import GPUConfig
import time

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class RouterManager:
    def __init__(self, model_client: ModelRpcClient, port_args: PortArgs):
        # Init communication
        context = zmq.asyncio.Context(6)
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{port_args.detokenizer_port}"
        )

        self.send_to_tokenizer = context.socket(zmq.PUSH)
        self.send_to_tokenizer.connect(f"tcp://127.0.0.1:{port_args.tokenizer_port}")
        
        self.send_to_migration_target = context.socket(zmq.PUSH)
        self.recv_from_migration_source = context.socket(zmq.PULL)
        self.recv_from_migration_source.bind(f'tcp://0.0.0.0:{port_args.migrate_port}')

        self.recv_from_sched = context.socket(zmq.PULL)
        self.recv_from_sched.connect(f"tcp://127.0.0.1:{port_args.router_port+10}")
        
        # Init status
        self.model_client = model_client
        self.recv_reqs = []

        # Init some configs
        self.extend_dependency_time = GLOBAL_BACKEND_CONFIG.extend_dependency_time
        
        # Dict[uid -> migration url]
        self.uid_to_migrate_decision: Dict[str, ReqState] = {}

    async def loop_for_forward(self):
        while True:
            next_step_input = list(self.recv_reqs)
            self.recv_reqs = []
            out_pyobjs = await self.model_client.step(next_step_input)

            for obj in out_pyobjs:
                await self.send_to_detokenizer.send_pyobj(obj)

            # async sleep for receiving the subsequent request and avoiding cache miss
            slept = False
            if len(out_pyobjs) != 0:
                has_finished = any([obj.finished for obj in out_pyobjs])
                if has_finished:
                    if self.extend_dependency_time > 0:
                        slept = True
                        await asyncio.sleep(self.extend_dependency_time)

            if not slept:
                await asyncio.sleep(0.0006)
            
            # await self.recv_from_sched.recv_pyobj()

    async def loop_for_push_request(self):
        while True:
            next_step_input = list(self.recv_reqs)
            self.recv_reqs = []
            for recv_req in next_step_input:
                await self.model_client.push_req_step(recv_req)
            await asyncio.sleep(0.0006)
    
    async def scheduler_metrics_request(self, recv_req: SchedulingMetricsReqInput):
        """
        Pipes the scheduler request model client to get the metrics back to detokenizer flow.

        Detokenizer used in order to follow structure of existing code.
        """
        start = time.time()
        waiting_time_tokenizer_manager = start - recv_req.tokenizer_dispatch_time
        out = await self.model_client.scheduler_metrics_request(recv_req)
        # out = await asyncio.to_thread(self.model_client.model_server.exposed_scheduler_metrics_request, recv_req)
        inner_time = time.time() - start
        out.waiting_time_tokenizer_manager = waiting_time_tokenizer_manager
        out.inner_router_time = inner_time
        out.manager_recv_time = recv_req.manager_recv_time
        out.manager_dispatch_time = time.time()
        await self.send_to_tokenizer.send_pyobj(out)

    async def loop_for_recv_requests(self, loop):
        """
        Recieves from tokenizer and forwards to model. 

        In the case that it's a scheduling metric request, it will be handled asynchronously
        """
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            if isinstance(recv_req, SchedulingMetricsReqInput):
                recv_req.manager_recv_time = time.time() - recv_req.tokenizer_dispatch_time
                loop.create_task(self.scheduler_metrics_request(recv_req))
                continue
            if isinstance(recv_req, str):
                loop.create_task(self.schedule_request_migration(recv_req))
                continue
            if isinstance(recv_req, DumpTrace):
                await self.dump_trace(recv_req)
                continue
            recv_req.append_to_queue_time = time.time()
            self.recv_reqs.append(recv_req)
    
    async def dump_trace(self, recv_req: DumpTrace):
        print(f"dumping trace to: {recv_req.fpath}")
        await self.model_client.dump_prefix_hit_trace(recv_req.fpath)
    
    async def loop_for_migration_requests(self, loop):
        while True:
            recv_req = await self.recv_from_migration_source.recv_pyobj()
            if isinstance(recv_req, MigrationReq):
                self.solve_migration_request(recv_req)
                continue
            raise ValueError(f"Invalid object: {recv_req}")
    
    #TODO: add actual migration logic
    #  1. Triggered periodically and internally
    #  2. Ask global scheduler for migration destination
    async def schedule_request_migration(self, url: str):
        self.send_to_migration_target.connect(url)
        candidates = await self.model_client.get_migrate_candidates()
        print(f"sending candidates: {candidates}")
        await self.send_to_migration_target.send_pyobj(MigrationReq(candidates))
        self.send_to_migration_target.disconnect(url)

    def solve_migration_request(self, mreq: MigrationReq):
        if mreq.requets:
            print(f"recving requests: {mreq.requets}")
            self.model_client.model_server.forward_queue.extend(mreq.requets)

def start_router_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    gpu_config: GPUConfig = None,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        model_client = ModelRpcClient(server_args, port_args, gpu_config=gpu_config)
        router = RouterManager(model_client, port_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_recv_requests(loop))
    loop.create_task(router.loop_for_migration_requests(loop))
    if server_args.freeze:
        loop.run_until_complete(router.loop_for_push_request())
    else:
        loop.run_until_complete(router.loop_for_forward())
    logging.info(f"Scheduling waiting overhead(s): {[model_client.model_server.schedule_waiting_overhead]}")