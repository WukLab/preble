from dataclasses import dataclass
from functools import lru_cache
import numpy as np

@dataclass(repr=False)
class Batch:
    reqs: list
    req_to_token_pool: dict
    token_to_kv_pool: dict
    tree_cache: dict

    # batched arguments to model runner
    input_ids: np.ndarray = None
    req_pool_indices: np.ndarray = None
    seq_lens: np.ndarray = None
    prefix_lens: np.ndarray = None
    position_ids_offsets: np.ndarray = None
    out_cache_loc: np.ndarray = None
    out_cache_cont_start: np.ndarray = None
    out_cache_cont_end: np.ndarray = None
    return_logprob: bool = False
    num_decoding_inputs: int = 0
    multiplex_extend_decode: bool = False

    # for multimodal
    pixel_values: list = None
    image_sizes: list = None
    image_offsets: list = None

    # other arguments for control
    output_ids: np.ndarray = None
    extend_num_tokens: int = None

    # batched sampling params
    temperatures: np.ndarray = None
    top_ps: np.ndarray = None
    top_ks: np.ndarray = None
    frequency_penalties: np.ndarray = None
    presence_penalties: np.ndarray = None
    logit_bias: np.ndarray = None

def llama2_7b_A6000_vllm(batch: Batch):
    num_batched_tokens = batch.input_ids.shape[0]
    num_attention_tokens = batch.seq_lens.sum()
    
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
        forward_time = -118 + 1.25 * num_batched_tokens - 2.56e-3 * num_batched_tokens**2
    else:
        forward_time = 22
    return forward_time / 1e3

def mistral_7b_A100_sglang_linear(num_batched_tokens: int):
    if num_batched_tokens >= 256:
        forward_time = 0.05031337 * num_batched_tokens + 5.505996896412796
    elif num_batched_tokens >= 64:
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
    seq_lens: np.ndarray = None,
):
    base = mistrial_7b_A6000_sglang_base(num_reqs, num_batched_tokens, total_context, num_unique_kv)
    attn_quad = 0
    for i, extend_lengths in enumerate(input_id_lens):
        if seq_lens is None:
            if extend_lengths >= 4096:
                attn_quad += -7.37 + 3.86e-3 * extend_lengths + 2.16e-6 * extend_lengths**2
        else:
            seq_len = seq_lens[i]
            if extend_lengths * seq_len > 1024 * 1024:
                attn_quad += 1.13e-3 * extend_lengths + 1.75e-3 * seq_len + 2.19e-6 * extend_lengths * seq_len
    attn_quad /= 1e3
    return (base + attn_quad) / 0.9

def llama3_70b_A100_tp2_sglang_extend_flashinfer(
    num_reqs, 
    num_batched_tokens, 
    total_context, 
    input_id_lens,
    num_unique_kv = None,
    seq_lens: np.ndarray = None,
):
    total_time = 0
    for i, extend_lengths in enumerate(input_id_lens):
        seq_len = seq_lens[i]
        total_time += 28.931389307785594 + 1.82233431e-01 * extend_lengths + 4.00365142e-03 * seq_len + 2.55050069e-06 * extend_lengths * seq_len
    total_time /= 1e3
    return total_time

def mistrial_7b_A6000_sglang_decode_flashinfer(
    num_reqs, 
    num_batched_tokens, 
    total_context, 
    num_unique_kv = None
):
    return mistrial_7b_A6000_sglang_base(num_reqs, num_batched_tokens, total_context, num_unique_kv) / 0.9

@lru_cache(maxsize=None)
def LP_mistral_7b_A6000_sglang_extend_flashinfer(num_extend_tokens, total_context):
    return mistral_7b_A6000_sglang_extend_flashinfer(
        1, num_extend_tokens, total_context, [num_extend_tokens], num_extend_tokens, np.array([total_context])
    )

def mistral_7b_A100_sglang_extend_flashinfer(
    num_reqs, 
    num_batched_tokens, 
    total_context, 
    input_id_lens,
    num_unique_kv = None,
    seq_lens: np.ndarray = None,
):
    linear = mistral_7b_A100_sglang_linear(num_batched_tokens)
    attn = 0
    for i, extend_lengths in enumerate(input_id_lens):
        seq_len = seq_lens[i]
        if extend_lengths * seq_len >= 1024 * 1024:
            attn += 0.12386358316396695 + -7.45358651e-04 * extend_lengths + 1.72022673e-03 * seq_len + 1.36265841e-06 * extend_lengths * seq_len
        else:
            attn += mistral_7b_A6000_sglang_attention(num_reqs, total_context, num_unique_kv)
    attn /= 1e3
    return (linear + attn) / 0.85

if __name__ == '__main__':
    test_extend = mistral_7b_A6000_sglang_extend_flashinfer(
        1, 512, 4097, [512], 7218, np.array([4097])
    )
    print(test_extend * 1000)
    
    test_mix = mistral_7b_A6000_sglang_extend_flashinfer(
        13, 512, 87722, [1]*12 + [512], 79629, np.array([7267]*12 + [7000])
    )
    print(test_mix * 1000)
