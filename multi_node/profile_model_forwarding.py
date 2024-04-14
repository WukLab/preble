import argparse
import logging
import random
import numpy as np
import uuid
import torch

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import handle_port_init
from sglang.srt.managers.router.model_rpc import ModelRpcClient, ModelRpcServer
from sglang.srt.managers.router.infer_batch import Batch
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

def get_sampling_params(max_new_tokens, tokenizer):
    sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
    if sampling_params.max_new_tokens != 0:
        sampling_params.normalize(tokenizer)
        sampling_params.verify()
    return sampling_params

def profile_prefill(model_client: ModelRpcClient, num_mul, num_prompt, ctx_len, token_id_start, num_prefix):
    model_server = model_client.model_server
    sampling_params = get_sampling_params(1, model_server.tokenizer)
    pixel_values, image_hash, image_size = None, None, None
    for i in range(num_mul + num_prompt):
        if i < num_mul:
            input_ids = [token_id_start + i % num_prefix] * ctx_len
        else:
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
            arrival_time=0.0,
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

def run_to_complete(model_client: ModelRpcClient, num_seqs, ctx_len, token_id_start, max_new_tokens, num_prefix):
    model_server = model_client.model_server
    sampling_params = get_sampling_params(max_new_tokens, model_server.tokenizer)
    pixel_values, image_hash, image_size = None, None, None
    inflight = set()
    for i in range(num_seqs):
        input_ids = [token_id_start + i % num_prefix] * ctx_len
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
            arrival_time=0.0,
        )
        inflight.add(tokenized_obj.rid)
        model_server.handle_generate_request(tokenized_obj)
        
    while inflight:
        model_server.forward_step()
        for output in model_server.out_pyobjs:
            for rid in output.rids:
                inflight.remove(rid)
        model_server.out_pyobjs = []

def run_to_scheduled(model_client: ModelRpcClient, num_seqs, starting_ctx_len, token_id_start, num_prefix):
    model_server = model_client.model_server
    sampling_params = get_sampling_params(starting_ctx_len, model_server.tokenizer)
    pixel_values, image_hash, image_size = None, None, None
    reqs = []
    for i in range(num_seqs):
        input_ids = [token_id_start + i % num_prefix] * starting_ctx_len
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
            arrival_time=0.0,
        )
        reqs.append(tokenized_obj)
        
    while True:
        model_server.handle_generate_request(reqs.pop())
        model_server.forward_step()
        if model_server.out_pyobjs:
            print(len(model_server.out_pyobjs[-1].rids))
        if model_server.out_pyobjs and len(model_server.out_pyobjs[-1].rids) == num_seqs:
            break
        model_server.out_pyobjs = []

def profile_multi_query(model_client: ModelRpcClient, num_prompt, num_mul, ctx_len, num_qs, token_id_start, num_prefix):
    if not num_prefix:
        num_prefix = num_mul
    if num_mul > 0:
        cached_ctx_len = ctx_len - num_qs
        run_to_complete(model_client, num_mul, cached_ctx_len, token_id_start, 1, num_prefix)
    torch.cuda.cudart().cudaProfilerStart()
    forward_time = profile_prefill(model_client, num_mul, num_prompt, ctx_len, token_id_start, num_prefix)
    torch.cuda.cudart().cudaProfilerStop()
    return forward_time

def profile_decoding(model_client: ModelRpcClient, num_seqs, ctx_len, token_id_start, num_prefix):
    model_server = model_client.model_server
    if not num_prefix:
        num_prefix = num_seqs
    run_to_complete(model_client, num_seqs, ctx_len, token_id_start, 1, num_prefix)
    
    sampling_params = get_sampling_params(2, model_server.tokenizer)
    pixel_values, image_hash, image_size = None, None, None
    for i in range(num_seqs):
        input_ids = [token_id_start + i % num_prefix] * ctx_len
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
            arrival_time=0.0
        )
        model_server.handle_generate_request(tokenized_obj)
    
    model_server.forward_step()
    num_stepped = sum(len(output.rids) for output in model_server.out_pyobjs)
    assert num_stepped == num_seqs, f"Expected {num_seqs} outputs, got {num_stepped}"
    model_server.out_pyobjs = []

    torch.cuda.cudart().cudaProfilerStart() 
    forward_time_events = model_server.forward_step()
    torch.cuda.cudart().cudaProfilerStop()
    num_stepped = sum(len(output.rids) for output in model_server.out_pyobjs)
    assert num_stepped == num_seqs, f"Expected {num_seqs} outputs, got {num_stepped}"
    model_server.out_pyobjs = []
    
    forward_time = 0
    for start, end in forward_time_events:
        end.synchronize()
        forward_time += start.elapsed_time(end)
    return forward_time
    
    
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
    warm_up_prompts = 16
    run_to_complete(model_client, warm_up_prompts, 2048, 0, 64, warm_up_prompts)
    torch.cuda.synchronize()
    if args.num_gen > 0:
        forward_time = profile_decoding(model_client, args.num_gen, args.ctx_len, warm_up_prompts, args.num_prefix)
        logging.info(f"Normal Decoding: {args.num_gen} x 1: {forward_time:.2f} ms")
        warm_up_prompts += args.num_gen
    if args.num_prompt + args.num_mul > 0:
        forward_time = profile_multi_query(model_client, args.num_prompt, args.num_mul, args.ctx_len, args.mul_qs, warm_up_prompts, args.num_prefix)
        logging.info(f"Multi Query Decoding: {args.num_prompt} x {args.ctx_len} + {args.num_mul} x {args.mul_qs}: {forward_time:.2f} ms")
        warm_up_prompts += args.num_mul + args.num_prompt
    

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
    parser.add_argument('--num-prefix', type=int, default=0,
                        help="Number of prefix patterns. 0 means no shared prefix")
    parser.add_argument('--ctx-len', type=int, default=128)
    parser.add_argument('--mul-qs', type=int, default=16)
    
    args = parser.parse_args()
    num_seqs = args.num_prompt + args.num_gen + args.num_mul
    print(f'num sequences: {num_seqs}\n',
             f'context length: {args.ctx_len}\n',
          f'num prompt: {args.num_prompt}\n',
          f'num gen: {args.num_gen}\n',
          f'num mul: {args.num_mul}\n',
          f'num prefix: {args.num_prefix}\n'
          f'queries per mul: {args.mul_qs}')
    # prompt + mul cannot be scheduled together with gen
    if args.num_prompt + args.num_mul > 0 and args.num_gen > 0:
        raise ValueError("Prompt and Multi Query cannot be scheduled together with normal decoding.")
    if args.num_gen > 0 and not args.stream_interval:
        args.stream_interval = 1
    main(args)