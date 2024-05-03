from sglang.srt.managers.router.infer_batch import Batch
import torch

"""Principles
Linear layer consider batching only
Attention:
    Long prompt: compute bound -> quadratic to context, each request independent
    Short decoding: memory bound -> linear to total context of all requests
                    consider shared prefix -> need to be improved
    Long decoding: memory bound but consider parallelism on sequence dimension
                        calculate long request independently with different linear scope
"""

def llama2_7b_A6000_vllm(batch: Batch):
    num_batched_tokens = batch.input_ids.shape[0]
    num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
    
    if num_batched_tokens >= 384:
        forward_time = 0.131*num_batched_tokens + 5.67
    elif num_batched_tokens >= 128:
        forward_time = 0.114*num_batched_tokens + 12.4
    else:
        forward_time = 26.06523603
    forward_time += num_attention_tokens / 2048 * 1.663659159
    forward_time /= 1e3 # to seconds
    return forward_time


def mistral_7b_A6000_sglang_linear(num_batched_tokens: int):
    if num_batched_tokens >= 384:
        forward_time = 0.10842571 * num_batched_tokens + 4.209777054806409
    elif num_batched_tokens >= 192:
    #    -118 + 1.25x + -2.56E-03x^2 
        forward_time = -118 + 1.25 * num_batched_tokens - 2.56e-3 * num_batched_tokens**2
    else:
        forward_time = 22
    return forward_time / 1e3

def mistral_7b_A100_sglang_linear(num_batched_tokens: int):
    if num_batched_tokens >= 256:
        forward_time = 0.05031337 * num_batched_tokens + 5.505996896412796
    elif num_batched_tokens >= 64:
    #    -118 + 1.25x + -2.56E-03x^2 
        forward_time = 10.5
    else:
        forward_time = 8.96
    return forward_time / 1e3


def mistral_7b_A6000_sglang_attention(num_reqs, total_context, num_unique_kv: int):
    if num_unique_kv is None:
        num_unique_kv = total_context
    if total_context <= 1024:
        forward_time = 0.32
    else:
        # 1.86E-04*x + 0.159
        forward_time = 1.86e-4 * total_context + 0.159
        if num_unique_kv / num_reqs <= 1024 and num_reqs * num_unique_kv <= 32 * 256 * 2048:
            forward_time /= 2
    return forward_time / 1e3

    
def mistrial_7b_A6000_sglang_base(num_reqs, num_batched_tokens, total_context, num_unique_kv = None):
    forward_time = mistral_7b_A6000_sglang_linear(num_batched_tokens) + \
                   mistral_7b_A6000_sglang_attention(num_reqs, total_context, num_unique_kv)
    return forward_time 

def mistral_7b_A6000_sglang_extend_flashinfer(
    num_reqs, 
    num_batched_tokens, 
    total_context, 
    input_id_lens,
    num_unique_kv = None,
    seq_lens: torch.Tensor = None,
):
    base = mistrial_7b_A6000_sglang_base(num_reqs, num_batched_tokens, total_context, num_unique_kv)
    attn_quad = 0
    for i, extend_lengths in enumerate(input_id_lens):
        if seq_lens is None:
            if extend_lengths >= 4096:
                #  -7.37 + 3.86E-03x + 2.16E-06x^2
                attn_quad += -7.37 + 3.86e-3 * extend_lengths + 2.16e-6 * extend_lengths**2
        else:
            seq_len = seq_lens[i].item()
            if extend_lengths * seq_len > 1024 * 1024:
                attn_quad += 1.13e-3 * extend_lengths + 1.75e-3 * seq_len + 2.19e-6 * extend_lengths * seq_len
    attn_quad /= 1e3
    return (base + attn_quad) / 0.9


def LP_mistral_7b_A6000_sglang_extend_flashinfer(num_extend_tokens, total_context):
    # if num_extend_tokens < 192:
    #     print("Warning: identify short node and not is_leaf, this node might add too much recompute cost")
    #     return num_extend_tokens / 1000
    return mistral_7b_A6000_sglang_extend_flashinfer(
        1, num_extend_tokens, total_context, [num_extend_tokens], num_extend_tokens, torch.tensor([total_context])
    )
    
def mistral_7b_A100_sglang_extend_flashinfer(
    num_reqs, 
    num_batched_tokens, 
    total_context, 
    input_id_lens,
    num_unique_kv = None,
    seq_lens: torch.Tensor = None,
):
    linear = mistral_7b_A100_sglang_linear(num_batched_tokens)
    attn = 0
    for i, extend_lengths in enumerate(input_id_lens):
        seq_len = seq_lens[i].item()
        if extend_lengths * seq_len >= 1024 * 1024:
            attn += 0.12386358316396695 + -7.45358651e-04 * extend_lengths + 1.72022673e-03 * seq_len + 1.36265841e-06 * extend_lengths * seq_len
        else:
            # This is insignificant
            attn += mistral_7b_A6000_sglang_attention(num_reqs, total_context, num_unique_kv)
    attn /= 1e3
    return (linear + attn) / 0.85


if __name__ == '__main__':
    test_extend = mistral_7b_A6000_sglang_extend_flashinfer(
        1, 512, 4097, [512], 7218, torch.Tensor([4097])
    )
    print(test_extend * 1000)
    
    test_mix = mistral_7b_A6000_sglang_extend_flashinfer(
        13, 512, 87722, [1]*12 + [512], 79629, torch.Tensor([7267]*12 + [7000])
    )
    print(test_mix * 1000)