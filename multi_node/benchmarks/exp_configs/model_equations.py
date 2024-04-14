from sglang.srt.managers.router.infer_batch import Batch

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
        forward_time = 0.115 * num_batched_tokens - 1.13
    elif num_batched_tokens >= 192:
    #    -118 + 1.25x + -2.56E-03x^2 
        forward_time = -118 + 1.25 * num_batched_tokens - 2.56e-3 * num_batched_tokens**2
    else:
        forward_time = 22
    return forward_time / 1e3


def mistral_7b_A6000_sglang_attention(num_reqs, num_attention_tokens, num_unique_kv: int):
    total_context = num_attention_tokens.sum()
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
    
def mistrial_7b_A6000_sglang_base(num_reqs, num_batched_tokens, num_attention_tokens, num_unique_kv = None):
    forward_time = mistral_7b_A6000_sglang_linear(num_batched_tokens) + \
                   mistral_7b_A6000_sglang_attention(num_reqs, num_attention_tokens, num_unique_kv)
    return forward_time 

def mistral_7b_A6000_sglang_extend_flashinfer(
    num_reqs, 
    num_batched_tokens, 
    num_attention_tokens, 
    input_id_lens, 
    num_unique_kv = None
):
    base = mistrial_7b_A6000_sglang_base(num_reqs, num_batched_tokens, num_attention_tokens, num_unique_kv)
    attn_quad = 0
    for extend_lengths in input_id_lens:
        if extend_lengths >= 4096:
            #  -7.37 + 3.86E-03x + 2.16E-06x^2
            attn_quad += -7.37 + 3.86e-3 * extend_lengths + 2.16e-6 * extend_lengths**2
    attn_quad /= 1e3
    return (base + attn_quad) / 0.95

def mistrial_7b_A6000_sglang_decode_flashinfer(
    num_reqs, 
    num_batched_tokens, 
    num_attention_tokens, 
    num_unique_kv = None
):
    return mistrial_7b_A6000_sglang_base(num_reqs, num_batched_tokens, num_attention_tokens, num_unique_kv) / 0.95


def LP_mistral_7b_A6000_sglang_extend_flashinfer(num_extend_tokens, is_leaf):
    if num_extend_tokens < 192 and not is_leaf:
        print("Warning: identify short node and not is_leaf, this node might add too much recompute cost")
    return mistral_7b_A6000_sglang_extend_flashinfer(
        1, num_extend_tokens, num_extend_tokens, [num_extend_tokens], num_extend_tokens
    )