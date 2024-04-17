from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import logging

import numpy as np
import torch
from sglang.srt.managers.router.radix_cache import RadixCache
from sglang.srt.memory_pool import ReqToTokenPool, TokenToKVPool

logger = logging.getLogger(__name__)


class ForwardMode(Enum):
    PREFILL = auto()
    EXTEND = auto()
    DECODE = auto()


class FinishReason(Enum):
    LENGTH = auto()
    EOS_TOKEN = auto()
    STOP_STR = auto()

class Req:
    def __init__(self, rid, input_text, input_ids, arrival_time, append_to_queue_time):
        self.rid = rid
        self.input_text = input_text
        self.input_ids = input_ids
        self.output_ids = []
        self.arrival_time = arrival_time
        self.append_to_queue_time = append_to_queue_time

        # Since jump forward may retokenize the prompt with partial outputs,
        # we maintain the original prompt length to report the correct usage.
        self.prompt_tokens = len(input_ids)
        # The number of decoded tokens for token usage report. Note that
        # this does not include the jump forward tokens.
        self.completion_tokens_wo_jump_forward = 0

        # For vision input
        self.pixel_values = None
        self.image_size = None
        self.image_offset = 0
        self.pad_value = None

        self.sampling_params = None
        self.return_logprob = False
        self.logprob_start_len = 0
        self.stream = False

        self.tokenizer = None
        self.finished = False
        self.finish_reason = None
        self.hit_stop_str = None

        self.extend_input_len = 0
        self.prefix_indices = []
        self.last_node = None

        self.logprob = None
        self.token_logprob = None
        self.normalized_logprob = None

        # For constrained decoding
        self.regex_fsm = None
        self.regex_fsm_state = 0
        self.jump_forward_map = None
        self.output_and_jump_forward_str = ""
        
        # For chunk-prefill
        self.num_computed_tokens = 0
        self.num_inflight_tokens = 0

    def max_new_tokens(self):
        return self.sampling_params.max_new_tokens

    def jump_forward_and_retokenize(self, jump_forward_str, next_state):
        old_output_str = self.tokenizer.decode(self.output_ids)
        # FIXME: This logic does not really solve the problem of determining whether
        # there should be a leading space.
        first_token = self.tokenizer.convert_ids_to_tokens(self.output_ids[0])
        first_token = (
            first_token.decode() if isinstance(first_token, bytes) else first_token
        )
        if first_token.startswith("▁"):
            old_output_str = " " + old_output_str
        new_input_string = (
            self.input_text
            + self.output_and_jump_forward_str
            + old_output_str
            + jump_forward_str
        )
        new_input_ids = self.tokenizer.encode(new_input_string)
        if self.pixel_values is not None:
            # NOTE: This is a hack because the old input_ids contains the image padding
            jump_forward_tokens_len = len(self.tokenizer.encode(jump_forward_str))
        else:
            jump_forward_tokens_len = (
                len(new_input_ids) - len(self.input_ids) - len(self.output_ids)
            )

        # print("=" * 100)
        # print(f"Catch jump forward:\n{jump_forward_str}")
        # print(self.tokenizer.convert_ids_to_tokens(self.input_ids))
        # print(self.tokenizer.convert_ids_to_tokens(new_input_ids))

        self.input_ids = new_input_ids
        self.output_ids = []
        self.sampling_params.max_new_tokens = max(
            self.sampling_params.max_new_tokens - jump_forward_tokens_len, 0
        )
        self.regex_fsm_state = next_state
        self.output_and_jump_forward_str = (
            self.output_and_jump_forward_str + old_output_str + jump_forward_str
        )

        # print(f"Output and jump forward str:\n{self.output_and_jump_forward_str}")
        # print("*" * 100)
    
    def check_finished(self):
        if self.finished:
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished = True
            self.finish_reason = FinishReason.LENGTH
            return

        if (
            self.output_ids[-1] == self.tokenizer.eos_token_id
            and self.sampling_params.ignore_eos == False
        ):
            self.finished = True
            self.finish_reason = FinishReason.EOS_TOKEN
            return

        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str:
                    self.finished = True
                    self.finish_reason = FinishReason.STOP_STR
                    self.hit_stop_str = stop_str
                    return
                
    def get_num_unfinished_tokens(self):
        return self.prompt_tokens + len(self.output_ids) - len(self.prefix_indices) - self.num_computed_tokens
    
    # NOTE: Currently sglang clears output tokens when recompute (??)
    #       so a prefill chunk will never involve output tokens
    #       Change this function if this is nolonger true
    def get_inflight_token_ids(self) -> List[int]:
        # logger.debug(f"num_computed_tokens={self.num_computed_tokens}, num_inflight_tokens={self.num_inflight_tokens}, prompt_len={self.prompt_tokens}, output_len={len(self.output_ids)}")
        start_idx = len(self.prefix_indices) + self.num_computed_tokens
        if start_idx >= self.prompt_tokens:
            assert self.num_computed_tokens + len(self.prefix_indices) == self.prompt_tokens + len(self.output_ids) - 1
            assert self.num_inflight_tokens == 1
            return [self.output_ids[-1]]
        return self.input_ids[start_idx : start_idx + self.num_inflight_tokens]
    
    def get_context_len(self):
        return len(self.prefix_indices) + self.num_computed_tokens + self.num_inflight_tokens
    
    def update_after_step(self):
        self.num_computed_tokens += self.num_inflight_tokens
        self.num_inflight_tokens = 0
    
    def reset_state(self):
        self.prefix_indices = None
        self.last_node = None
        self.extend_input_len = 0
        self.output_ids = []
        self.regex_fsm_state = 0
        self.num_computed_tokens = 0

    def __repr__(self):
        return f"rid(n={self.rid}, " f"input_ids={self.input_ids}, "


@dataclass
class SchedulingBudget:
    max_new_tokens: int
    scheduled_tokens: int
    
    def get_remaining_token_budget(self):
        return self.max_new_tokens - self.scheduled_tokens

    def schedule_new_tokens(self, num_tokens):
        assert num_tokens <= self.get_remaining_token_budget()
        self.scheduled_tokens += num_tokens


@dataclass
class Batch:
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: TokenToKVPool
    tree_cache: RadixCache

    # batched arguments to model runner
    input_ids: torch.Tensor = None
    req_pool_indices: torch.Tensor = None
    seq_lens: torch.Tensor = None
    prefix_lens: torch.Tensor = None
    position_ids_offsets: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
    out_cache_cont_start: torch.Tensor = None
    out_cache_cont_end: torch.Tensor = None
    return_logprob: bool = False

    # for multimodal
    pixel_values: List[torch.Tensor] = None
    image_sizes: List[List[int]] = None
    image_offsets: List[int] = None

    # other arguments for control
    output_ids: torch.Tensor = None
    extend_num_tokens: int = None

    # batched sampling params
    temperatures: torch.Tensor = None
    top_ps: torch.Tensor = None
    top_ks: torch.Tensor = None
    frequency_penalties: torch.Tensor = None
    presence_penalties: torch.Tensor = None
    logit_bias: torch.Tensor = None
    
    # for simulator
    input_id_lengths: List[int] = None

    @classmethod
    def init_new(cls, reqs, req_to_token_pool, token_to_kv_pool, tree_cache):
        return_logprob = any(req.return_logprob for req in reqs)

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            tree_cache=tree_cache,
            return_logprob=return_logprob,
        )

    def is_empty(self):
        return len(self.reqs) == 0

    def prepare_for_extend(self, vocab_size: int, int_token_logit_bias: torch.Tensor):
        device = "cuda"
        bs = len(self.reqs)
        reqs = self.reqs
        input_ids = [r.input_ids[len(r.prefix_indices) :] for r in reqs]
        prefix_indices = [r.prefix_indices for r in reqs]

        # Handle prefix
        flatten_input_ids = []
        extend_lens = []
        prefix_lens = []
        seq_lens = []

        req_pool_indices = self.req_to_token_pool.alloc(bs)
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        for i in range(bs):
            flatten_input_ids.extend(input_ids[i])
            extend_lens.append(len(input_ids[i]))

            if len(prefix_indices[i]) == 0:
                prefix_lens.append(0)
            else:
                prefix_lens.append(len(prefix_indices[i]))
                self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
                    : len(prefix_indices[i])
                ] = prefix_indices[i]

            seq_lens.append(prefix_lens[-1] + extend_lens[-1])
            self.reqs[i].num_inflight_tokens = extend_lens[-1]

        position_ids_offsets = torch.zeros((bs,), dtype=torch.int32, device=device)
        self.input_id_lengths = extend_lens
        
        # Alloc mem
        seq_lens, prefix_lens = np.array(seq_lens), np.array(prefix_lens)
        extend_num_tokens = seq_lens.sum() - prefix_lens.sum()
        out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)
        if out_cache_loc is None:
            if not self.tree_cache.disable:
                self.tree_cache.evict(extend_num_tokens, self.token_to_kv_pool.free)
                out_cache_loc = self.token_to_kv_pool.alloc(extend_num_tokens)

            if out_cache_loc is None:
                print("Prefill out of memory. This should nerver happen.")
                self.tree_cache.pretty_print()
                exit()

        pt = 0
        for i in range(bs):
            self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
                prefix_lens[i] : prefix_lens[i] + extend_lens[i]
            ] = out_cache_loc[pt : pt + extend_lens[i]]
            pt += extend_lens[i]

        # Handle logit bias
        logit_bias = torch.zeros((bs, vocab_size), dtype=torch.float32, device=device)
        for i in range(bs):
            if reqs[i].sampling_params.dtype == "int":
                logit_bias[i] = int_token_logit_bias

        # Set fields
        self.input_ids = torch.tensor(
            flatten_input_ids, dtype=torch.int32, device=device
        )
        self.pixel_values = [r.pixel_values for r in reqs]
        self.image_sizes = [r.image_size for r in reqs]
        self.image_offsets = [
            r.image_offset - p_len for r, p_len in zip(reqs, prefix_lens)
        ]
        self.req_pool_indices = req_pool_indices
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        self.prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        self.position_ids_offsets = position_ids_offsets
        self.extend_num_tokens = extend_num_tokens
        self.out_cache_loc = out_cache_loc

        self.temperatures = torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        self.top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float, device=device
        ).view(-1, 1)
        self.top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int, device=device
        ).view(-1, 1)
        self.frequency_penalties = torch.tensor(
            [r.sampling_params.frequency_penalty for r in reqs],
            dtype=torch.float,
            device=device,
        )
        self.presence_penalties = torch.tensor(
            [r.sampling_params.presence_penalty for r in reqs],
            dtype=torch.float,
            device=device,
        )
        self.logit_bias = logit_bias

    def check_decode_mem(self):
        bs = len(self.reqs)
        if self.token_to_kv_pool.available_size() >= bs:
            return True

        if not self.tree_cache.disable:
            self.tree_cache.evict(bs, self.token_to_kv_pool.free)
        if self.token_to_kv_pool.available_size() >= bs:
            return True

        return False

    def retract_decode(self):
        sorted_indices = [i for i in range(len(self.reqs))]
        # sorted_indices.sort(
        #     key=lambda i: (len(self.reqs[i].output_ids), -len(self.reqs[i].input_ids)),
        #     reverse=True,
        # )
        # sorted_indices.sort(
        #     key=lambda i: (self.reqs[i].arrival_time, len(self.reqs[i].output_ids))
        # )
        sorted_indices.sort(
            key=lambda i: (self.reqs[i].arrival_time, len(self.reqs[i].output_ids))
        )

        retracted_reqs = []
        seq_lens_np = self.seq_lens.cpu().numpy()
        req_pool_indices_np = self.req_pool_indices.cpu().numpy()
        while self.token_to_kv_pool.available_size() < len(self.reqs):
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            self.tree_cache.dec_ref_counter(req.last_node)
            req.reset_state()

            # TODO: apply more fine-grained retraction

            token_indices = self.req_to_token_pool.req_to_token[
                req_pool_indices_np[idx]
            ][: seq_lens_np[idx]]
            self.token_to_kv_pool.free(token_indices)

        self.filter_batch(sorted_indices)

        return retracted_reqs


    def check_for_jump_forward(self):
        jump_forward_reqs = []
        filter_indices = [i for i in range(len(self.reqs))]

        req_pool_indices_cpu = None

        for i, req in enumerate(self.reqs):
            if req.jump_forward_map is not None:
                res = req.jump_forward_map.jump_forward(req.regex_fsm_state)
                if res is not None:
                    jump_forward_str, next_state = res
                    if len(jump_forward_str) <= 1:
                        continue

                    # insert the old request into tree_cache
                    token_ids_in_memory = tuple(req.input_ids + req.output_ids)[:-1]
                    if req_pool_indices_cpu is None:
                        req_pool_indices_cpu = self.req_pool_indices.cpu().tolist()
                    req_pool_idx = req_pool_indices_cpu[i]
                    indices = self.req_to_token_pool.req_to_token[
                        req_pool_idx, : len(token_ids_in_memory)
                    ]
                    prefix_len = self.tree_cache.insert(
                        token_ids_in_memory, indices.clone()
                    )
                    self.token_to_kv_pool.free(indices[:prefix_len])
                    self.req_to_token_pool.free(req_pool_idx)
                    self.tree_cache.dec_ref_counter(req.last_node)

                    # jump-forward
                    req.jump_forward_and_retokenize(jump_forward_str, next_state)

                    jump_forward_reqs.append(req)
                    filter_indices.remove(i)

        if len(filter_indices) < len(self.reqs):
            self.filter_batch(filter_indices)

        return jump_forward_reqs

    def prepare_for_decode(self, input_ids=None):
        # TODO: change to chunk-prefill
        if input_ids is None:
            input_ids = [
                r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs
            ]
        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        self.seq_lens.add_(1)
        self.prefix_lens = None
        self.input_id_lengths = [1] * len(input_ids)
        
        # Alloc mem
        bs = len(self.reqs)
        alloc_res = self.token_to_kv_pool.alloc_contiguous(bs)
        if alloc_res is None:
            self.out_cache_loc = self.token_to_kv_pool.alloc(bs)

            if self.out_cache_loc is None:
                print("Decode out of memory. This should nerver happen.")
                self.tree_cache.pretty_print()
                exit()

            self.out_cache_cont_start = None
            self.out_cache_cont_end = None
        else:
            self.out_cache_loc = alloc_res[0]
            self.out_cache_cont_start = alloc_res[1]
            self.out_cache_cont_end = alloc_res[2]

        self.req_to_token_pool.req_to_token[
            self.req_pool_indices, self.seq_lens - 1
        ] = self.out_cache_loc

    def filter_batch(self, unfinished_indices: List[int]):
        self.reqs = [self.reqs[i] for i in unfinished_indices]
        new_indices = torch.tensor(unfinished_indices, dtype=torch.int32, device="cuda")
        self.seq_lens = self.seq_lens[new_indices]
        self.input_ids = None
        self.req_pool_indices = self.req_pool_indices[new_indices]
        self.prefix_lens = None
        self.position_ids_offsets = self.position_ids_offsets[new_indices]
        self.out_cache_loc = self.out_cache_cont_start = self.out_cache_cont_end = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "frequency_penalties",
            "presence_penalties",
            "logit_bias",
        ]:
            setattr(self, item, getattr(self, item)[new_indices])

    def merge(self, other):
        self.reqs.extend(other.reqs)

        self.req_pool_indices = torch.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.prefix_lens = None
        self.position_ids_offsets = torch.concat(
            [self.position_ids_offsets, other.position_ids_offsets]
        )
        self.out_cache_loc = self.out_cache_cont_start = self.out_cache_cont_end = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "frequency_penalties",
            "presence_penalties",
            "logit_bias",
        ]:
            setattr(
                self, item, torch.concat([getattr(self, item), getattr(other, item)])
            )
    
    def _schedule_running(
        self, 
        budget: SchedulingBudget
    ) -> List[Req]:
        current_running_idx = sorted(
            [i for i in range(len(self.reqs))],
            key=lambda i: (self.reqs[i].arrival_time, len(self.reqs[i].output_ids))
        )
        req_pool_indices_np = self.req_pool_indices.cpu().numpy()
        preempted = []
        scheduled = []
        self.out_cache_loc = None
        while current_running_idx:
            # try to schedule each request
            target_idx = current_running_idx.pop(0)
            target_to_schedule = self.reqs[target_idx]
            # 1. check required new memory and evict if needed
            new_tokens = min(target_to_schedule.get_num_unfinished_tokens(), budget.get_remaining_token_budget())
            if (self.token_to_kv_pool.available_size() < new_tokens 
                and not self.tree_cache.disable):
                self.tree_cache.evict(new_tokens, self.token_to_kv_pool.free)
            # 2. if not enough, evict running requests
            while self.token_to_kv_pool.available_size() < new_tokens:
                if not current_running_idx:
                    evict_idx = target_idx
                    evict_req = target_to_schedule
                else:
                    evict_idx = current_running_idx.pop()
                    evict_req = self.reqs[evict_idx]
                    
                preempted.append(evict_req)
                self.tree_cache.dec_ref_counter(evict_req.last_node)
                evict_req.reset_state()
                token_indices = self.req_to_token_pool.req_to_token[
                    req_pool_indices_np[evict_idx]
                ][: evict_req.num_computed_tokens]
                self.token_to_kv_pool.free(token_indices)
                self.req_to_token_pool.free(req_pool_indices_np[evict_idx])
                
                # not enought memory to schedule
                if evict_idx == target_idx:
                    break
            else:
            #3. schedule it, allocate and set workload
                budget.schedule_new_tokens(new_tokens)
                scheduled.append(target_idx)
                out_cache_loc = self.token_to_kv_pool.alloc(new_tokens)
                self.req_to_token_pool.req_to_token[
                    req_pool_indices_np[target_idx],
                    target_to_schedule.num_computed_tokens : target_to_schedule.num_computed_tokens + new_tokens
                ] = out_cache_loc
                target_to_schedule.num_inflight_tokens = new_tokens
                if self.out_cache_loc is None:
                    self.out_cache_loc = out_cache_loc
                else:
                    self.out_cache_loc = torch.cat([self.out_cache_loc, out_cache_loc])
        if len(scheduled) < len(self.reqs):
            self.filter_batch(scheduled)
        self.out_cache_cont_start = self.out_cache_cont_end = None
        return preempted
    
    def prepare_for_inference(self):
        device = 'cuda'
        input_ids = [r.get_inflight_token_ids() for r in self.reqs]
        self.input_id_lengths = [len(ids) for ids in input_ids]
        input_ids = sum(input_ids, [])
        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        self.seq_lens = torch.tensor([r.get_context_len() for r in self.reqs], dtype=torch.int32, device=device)
        self.prefix_lens = None
        
            
    def sample(self, logits: torch.Tensor):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(self.temperatures)
        logits.add_(self.logit_bias)

        has_regex = any(req.regex_fsm is not None for req in self.reqs)
        if has_regex:
            allowed_mask = torch.empty_like(logits[0], dtype=torch.bool)
            for i, req in enumerate(self.reqs):
                if req.regex_fsm is not None:
                    allowed_mask.zero_()
                    allowed_mask[
                        req.regex_fsm.allowed_token_ids(req.regex_fsm_state)
                    ] = 1
                    logits[i].masked_fill_(~allowed_mask, float("-inf"))

        # TODO(lmzheng): apply penalty
        probs = torch.softmax(logits, dim=-1)
        probs_sort, probs_idx = _top_p_top_k(probs, self.top_ps, self.top_ks)
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(
            -1
        )
        batch_next_token_probs = torch.gather(
            probs_sort, dim=1, index=sampled_index
        ).view(-1)

        if has_regex:
            batch_next_token_ids_cpu = batch_next_token_ids.cpu().numpy()
            for i, req in enumerate(self.reqs):
                if req.regex_fsm is not None:
                    req.regex_fsm_state = req.regex_fsm.next_state(
                        req.regex_fsm_state, batch_next_token_ids_cpu[i]
                    )

        return batch_next_token_ids, batch_next_token_probs


def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps] = 0.0
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1) >= top_ks
    ] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    return probs_sort, probs_idx
