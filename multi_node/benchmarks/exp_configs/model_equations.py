from sglang.srt.managers.router.infer_batch import Batch

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

def mistral_7b_A6000_sglang_extend_flashinfer(batch: Batch):
    num_batched_tokens = batch.input_ids.shape[0]
    if num_batched_tokens >= 384:
        forward_time = 0.128 * num_batched_tokens + 9.97
    elif num_batched_tokens >= 192:
        forward_time = 0.117 * num_batched_tokens + 17.3
    else:
        forward_time = 30
    return forward_time / 1e3

def mistrial_7b_A6000_sglang_decode_flashinfer(batch: Batch):
    num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
    if num_attention_tokens >= 64 * 2048:
        forward_time = num_attention_tokens * 4e-4
    else:
        forward_time = 10.4 * num_attention_tokens**0.117
    return forward_time / 1e3

def mistral_7b_A6000_sglang_linear(num_batched_tokens: int):
    if num_batched_tokens >= 384:
        forward_time = 0.115 * num_batched_tokens - 1.13
    elif num_batched_tokens >= 192:
    #    -118 + 1.25x + -2.56E-03x^2 
        forward_time = -118 + 1.25 * num_batched_tokens - 2.56e-3 * num_batched_tokens**2
    else:
        forward_time = 22
    return forward_time / 1e3


def mistral_7b_A6000_sglang_attention(total_context: int, num_unique_kv: int):
    if total_context <= 32*1900:
        forward_time = 0.635 * (total_context ** 0.272)
    else:
        no_shared = total_context * 0.00024
        if num_unique_kv <= 32 * 1900:
            forward_time = no_shared / 2
        else:
            s = no_shared / 2 / (total_context - 32*1900)
            forward_time = s * (num_unique_kv - 32*1900) + no_shared / 2
    return forward_time / 1e3
    
def mistrial_7b_A6000_sglang_total(batch: Batch, num_unique_kv = None):
    num_batched_tokens = batch.input_ids.shape[0]
    num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
    if num_unique_kv is None:
        num_unique_kv = num_attention_tokens
    forward_time = mistral_7b_A6000_sglang_linear(num_batched_tokens) + \
                   mistral_7b_A6000_sglang_attention(num_attention_tokens, num_unique_kv)
    return forward_time / 0.9