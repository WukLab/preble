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

def sample_request(token_id, seq_len, max_new_tokens, stream, tokenizer):
    sampling_params = get_sampling_params(max_new_tokens, tokenizer)
    pixel_values, image_hash, image_size = None, None, None
    input_ids = [token_id] * seq_len
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
        stream=stream,
        arrival_time=0.0,
    )
    return tokenized_obj

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
    model_server = model_client.model_server
    
    # Warm up
    warm_up_prompts = 16
    warm_up_token_ids = max(token_ids) + 1
    run_to_complete(model_client, warm_up_prompts, 2048, warm_up_token_ids, 64, warm_up_prompts)
    torch.cuda.synchronize()
    
    # Add request to populate cache tree and kv cache
    prefix_lens = [seq_len - query_len for seq_len, query_len in zip(seq_lens, query_lens)]
    inflight = set()
    for prefix_len, token_id in zip(prefix_lens, token_ids):
        if prefix_len > 0:
            tokenized_obj = sample_request(token_id, prefix_len, 1, False, model_client.model_server.tokenizer)
            inflight.add(tokenized_obj.rid)
            model_server.handle_generate_request(tokenized_obj)
    while inflight:
        model_server.forward_step()
        for output in model_server.out_pyobjs:
            for rid in output.rids:
                inflight.remove(rid)
        model_server.out_pyobjs = []
    
    # Add actual workload
    for seq_len, token_id in zip(seq_lens, token_ids):
        tokenized_obj = sample_request(token_id, seq_len, 1, False, model_client.model_server.tokenizer)
        model_server.handle_generate_request(tokenized_obj)
        
    torch.cuda.cudart().cudaProfilerStart() 
    forward_time_events = model_server.forward_step()
    torch.cuda.cudart().cudaProfilerStop()
    num_finished_reqs = sum(len(output.rids) for output in model_server.out_pyobjs)
    assert num_finished_reqs == num_seqs, f"Expected {num_seqs} outputs, got {num_finished_reqs}"
    model_server.out_pyobjs = []
    
    forward_time = 0
    for start, end in forward_time_events:
        end.synchronize()
        forward_time += start.elapsed_time(end)
    print(f"Forward time: {forward_time:.3f} ms")    
    

if __name__ == "__main__":
    random.seed(10)
    np.random.seed(10)
    
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    if not args.stream_interval:
        args.stream_interval = 1
    if not args.chunk_prefill_budget or args.chunk_prefill_budget <= 1:
        raise ValueError("chunk_prefill_budget must be greater than 1")
        
    query_lens = [1] * 1
    seq_lens = [8192] * 1
    token_ids = [i for i in range(len(seq_lens))]
    
    num_seqs = len(seq_lens)
    print(f'total num seqs: {num_seqs}\n'
          f'total batched token: {sum(query_lens)}\n'
          f'total attention tokens: {sum(seq_lens)}\n'
          )
    main(args)