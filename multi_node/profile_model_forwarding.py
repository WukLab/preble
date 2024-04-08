import argparse
import logging
import random
import numpy as np
import uuid

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import handle_port_init
from sglang.srt.managers.router.model_rpc import ModelRpcClient, ModelRpcServer
from sglang.srt.managers.router.infer_batch import Batch
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

def profile_prefill(model_client: ModelRpcClient, num_seqs, ctx_len, token_id_start):
    model_server = model_client.model_server
    sampling_params = SamplingParams(max_new_tokens=1)
    pixel_values, image_hash, image_size = None, None, None
    for i in range(num_seqs):
        input_ids = [token_id_start + i] * ctx_len
        tokenized_obj = TokenizedGenerateReqInput(
            rid=uuid.uuid4().hex,
            input_text="",
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_hash=image_hash,
            image_size=image_size,
            sampling_params=sampling_params,
            return_logprob=None,
            logprob_start_len=None,
            stream=False,
        )
        model_server.handle_generate_request(tokenized_obj)
    forward_time_events = model_server.forward_step()
    
    num_finished_reqs = sum(len(output.rids) for output in model_server.out_pyobjs)
    assert num_finished_reqs == num_seqs, f"Expected {num_seqs} outputs, got {num_finished_reqs}"
    model_server.out_pyobjs = []
    
    forward_time = 0
    for start, end in forward_time_events:
        end.synchronize()
        forward_time += start.elapsed_time(end)
    return forward_time

def run_to_complete(model_client: ModelRpcClient, num_seqs, ctx_len, token_id_start):
    model_server = model_client.model_server
    sampling_params = SamplingParams(max_new_tokens=1)
    pixel_values, image_hash, image_size = None, None, None
    inflight = set()
    for i in range(num_seqs):
        input_ids = [token_id_start + i] * ctx_len
        tokenized_obj = TokenizedGenerateReqInput(
            rid=uuid.uuid4().hex,
            input_text="",
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_hash=image_hash,
            image_size=image_size,
            sampling_params=sampling_params,
            return_logprob=None,
            logprob_start_len=None,
            stream=False,
        )
        inflight.add(tokenized_obj.rid)
        model_server.handle_generate_request(tokenized_obj)
        
    while inflight:
        model_server.forward_step()
        for output in model_server.out_pyobjs:
            for rid in output.rids:
                inflight.remove(rid)
        model_server.out_pyobjs = []

def run_to_scheduled(model_client: ModelRpcClient, num_seqs, starting_ctx_len, token_id_start):
    model_server = model_client.model_server
    sampling_params = SamplingParams(max_new_tokens=starting_ctx_len)
    if sampling_params.max_new_tokens != 0:
        sampling_params.normalize(model_server.tokenizer)
        sampling_params.verify()
    pixel_values, image_hash, image_size = None, None, None
    for i in range(num_seqs):
        input_ids = [token_id_start + i] * starting_ctx_len
        tokenized_obj = TokenizedGenerateReqInput(
            rid=uuid.uuid4().hex,
            input_text="",
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_hash=image_hash,
            image_size=image_size,
            sampling_params=sampling_params,
            return_logprob=None,
            logprob_start_len=None,
            stream=True,
        )
        model_server.handle_generate_request(tokenized_obj)
        
    while True:
        model_server.forward_step()
        if model_server.out_pyobjs:
            print(len(model_server.out_pyobjs[-1].rids))
        if model_server.out_pyobjs and len(model_server.out_pyobjs[-1].rids) == num_seqs:
            break
        model_server.out_pyobjs = []

def profile_multi_query(model_client: ModelRpcClient, num_seqs, ctx_len, num_qs, token_id_start):
    cached_ctx_len = ctx_len - num_qs
    run_to_complete(model_client, num_seqs, cached_ctx_len, token_id_start)
    forward_time = profile_prefill(model_client, num_seqs, ctx_len, token_id_start)
    return forward_time

def profile_decoding(model_client: ModelRpcClient, num_seqs, ctx_len, token_id_start):
    run_to_scheduled(model_client, num_seqs, ctx_len, token_id_start)
    model_server = model_client.model_server
    forward_time_events = model_server.forward_step()
    assert model_server.out_pyobjs and len(model_server.out_pyobjs[-1].rids) == num_seqs
    forward_time = 0
    for start, end in forward_time_events:
        end.synchronize()
        forward_time += start.elapsed_time(end)
    return forward_time / len(forward_time_events)
    
    
def main(args):
    server_args = ServerArgs.from_cli_args(args)
    port, additional_ports = handle_port_init(server_args.port, server_args.additional_ports, server_args.tp_size)
    server_args.port = port
    server_args.additional_ports = additional_ports
    port_args = PortArgs(
        tokenizer_port=server_args.additional_ports[0],
        router_port=server_args.additional_ports[1],
        detokenizer_port=server_args.additional_ports[2],
        nccl_port=server_args.additional_ports[3],
        migrate_port=server_args.additional_ports[4],
        model_rpc_ports=server_args.additional_ports[5:],
    )
    model_client = ModelRpcClient(server_args, port_args)
    
    # Warm up
    warm_up_prompts = 4
    profile_prefill(model_client, warm_up_prompts, 2048, 0)
    
    if args.num_prompt > 0:
        forward_time = profile_prefill(model_client, args.num_prompt, args.ctx_len, warm_up_prompts)
        logging.info(f"Prefill: {args.num_prompt} x {args.ctx_len}: {forward_time:.2f} ms")
        warm_up_prompts += args.num_prompt
    if args.num_gen > 0:
        forward_time = profile_decoding(model_client, args.num_gen, args.ctx_len, warm_up_prompts)
        logging.info(f"Normal Decoding: {args.num_gen} x 1: {forward_time:.2f} ms")
        warm_up_prompts += args.num_gen
    if args.num_mul > 0:
        forward_time = profile_multi_query(model_client, args.num_mul, args.ctx_len, args.mul_qs, warm_up_prompts)
        logging.info(f"Multi Query Decoding: {args.num_mul} x {args.mul_qs}: {forward_time:.2f} ms")
        warm_up_prompts += args.num_mul
    

if __name__ == "__main__":
    random.seed(10)
    np.random.seed(10)
    
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--num-prompt", type=int, default=0,
                        help="Number of prompts to process.")
    parser.add_argument("--num-gen", type=int, default=0,
                        help="Number of normal decoding.")
    parser.add_argument("--num-mul", type=int, default=0,
                         help="Number of multi token decoding.")
    
    parser.add_argument('--ctx-len', type=int, default=128)
    parser.add_argument('--mul-qs', type=int, default=16)
    
    args = parser.parse_args()
    num_seqs = args.num_prompt + args.num_gen + args.num_mul
    print(f'num sequences: {num_seqs}\n',
             f'context length: {args.ctx_len}\n',
          f'num prompt: {args.num_prompt}\n',
          f'num gen: {args.num_gen}\n',
          f'num mul: {args.num_mul}\n',
          f'queries per mul: {args.mul_qs}')
    
    main(args)