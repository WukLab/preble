import asyncio
import logging
import multiprocessing
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import json
from dataclasses import dataclass
import math

import rpyc
import torch
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from sglang.srt.constrained.fsm_cache import FSMCache
from sglang.srt.constrained.jump_forward import JumpForwardCache
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.io_struct import (
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedGenerateReqInput,
    SchedulingMetricsReqInput, 
    SchedulingMetricsOut
)
from sglang.srt.managers.router.infer_batch import Batch, ForwardMode, Req, SchedulingBudget
from sglang.srt.managers.router.model_runner import ModelRunner
from sglang.srt.managers.router.radix_cache import RadixCache
from sglang.srt.managers.router.scheduler import Scheduler
from sglang.srt.model_config import ModelConfig
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.router.model_runner import GPUConfig
from sglang.srt.utils import (
    get_exception_traceback,
    get_int_token_logit_bias,
    is_multimodal_model,
    set_random_seed,
)
from vllm.logger import _default_handler as vllm_default_handler
from collections import deque
import os

logger = logging.getLogger("model_rpc")

detail_batch_logger = logger.debug

class ModelRpcServer:
    def __init__(
        self,
        tp_rank: int,
        server_args: ServerArgs,
        port_args: PortArgs,
        simulate: bool = False,
        gpu_config: GPUConfig = None,
    ):
        server_args, port_args = [obtain(x) for x in [server_args, port_args]]
        self.gpu_config = gpu_config
        self.chunk_prefill_budget = server_args.chunk_prefill_budget
        # self.use_sleep_forwarding = False if not gpu_config else gpu_config.forward_simulation is not None
        self.use_sleep_forwarding = False

        logger.info(f"Use sleep forwarding: {self.use_sleep_forwarding}")
        # Copy arguments
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.schedule_heuristic = server_args.schedule_heuristic
        logger.info(f"schedule_heuristic: {self.schedule_heuristic}")
        self.disable_regex_jump_forward = server_args.disable_regex_jump_forward
        vllm_default_handler.setLevel(
            level=getattr(logging, server_args.log_level.upper())
        )

        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
        )

        # for model end global settings
        server_args_dict = {
            "enable_flashinfer": server_args.enable_flashinfer,
            "attention_reduce_in_fp32": server_args.attention_reduce_in_fp32,
        }
        
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=port_args.nccl_port,
            load_format=server_args.load_format,
            trust_remote_code=server_args.trust_remote_code,
            server_args_dict=server_args_dict,
            simulate=simulate,
            gpu_config=gpu_config,
        )
        if is_multimodal_model(server_args.model_path):
            self.processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
        self.max_total_num_token = self.model_runner.max_total_num_token
        self.max_num_running_seq = self.max_total_num_token // 2
        self.max_prefill_num_token = max(
            self.model_config.context_len,
            (
                self.max_total_num_token // 6
                if server_args.max_prefill_num_token is None
                else server_args.max_prefill_num_token
            ),
        )
        self.int_token_logit_bias = torch.tensor(
            get_int_token_logit_bias(self.tokenizer, self.model_config.vocab_size)
        )
        set_random_seed(server_args.random_seed)
        logger.info(
            f"Rank {self.tp_rank}: "
            f"max_total_num_token={self.max_total_num_token}, "
            f"max_prefill_num_token={self.max_prefill_num_token}, "
            f"context_len={self.model_config.context_len}, "
        )
        logger.info(server_args.get_optional_modes_logging())

        # Init cache
        self.tree_cache = RadixCache(server_args.disable_radix_cache)
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.scheduler = Scheduler(
            self.schedule_heuristic,
            self.max_num_running_seq,
            self.max_prefill_num_token,
            self.max_total_num_token,
            self.tree_cache,
        )
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool = self.model_runner.token_to_kv_pool

        # Init running status
        self.forward_queue: List[Req] = []
        self.running_batch: Batch = None
        self.delayed_batch : Batch = None
        self.multi_priority_queue: List[List[Req]] = [[] for _ in range(10)]
        self.start_schedule_from = 9

        # Store the length and running batch sizes in a buffer since they variance is noisy
        self.forward_queue_len_buffer = deque(maxlen=server_args.metrics_buffer_size)
        self.running_batch_len_buffer = deque(maxlen=server_args.metrics_buffer_size)
        self.hit_trace_buffer = deque()
        self.hit_trace_window_size = server_args.hit_trace_window_size

        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval

        # Init the FSM cache for constrained generation
        self.regex_fsm_cache = FSMCache(
            server_args.tokenizer_path,
            {
                "tokenizer_mode": server_args.tokenizer_mode,
                "trust_remote_code": server_args.trust_remote_code,
            },
        )
        self.jump_forward_cache = JumpForwardCache()

        # Init new token estimation
        self.new_token_ratio = min(0.4 * server_args.schedule_conservativeness, 1.0)
        self.min_new_token_ratio = min(0.2 * server_args.schedule_conservativeness, 1.0)
        self.new_token_ratio_step = (0.0001, 0.05)  # (down, up)
        self.log_prefix_hit = server_args.log_prefix_hit
        self.prefix_hit_trace = []
        if not self.gpu_config:
            self.current_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[torch.cuda.current_device()]
        else:
            self.current_gpu = self.gpu_config.gpu_id
        self.total_scheduling_overhead = 0
        self.schedule_waiting_overhead = 0
        self.recomputed_tokens = 0
        self.total_forwarded_tokens = 0
        self.total_cache_hit_tokens = 0
        self.iter_cnt = -1

    def flush_cache(self):
        if self.num_waiting_reqs() == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            self.regex_fsm_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
        else:
            warnings.warn(
                "Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.forward_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )
    
    def num_waiting_reqs(self):
        return len(self.forward_queue) + sum(len(pg) for pg in self.multi_priority_queue)
    
    def update_hit_trace(self, timestamp, hit_tokens, total_prompt_len):
        self.hit_trace_buffer.append((timestamp, hit_tokens, total_prompt_len))
        while self.hit_trace_buffer and timestamp - self.hit_trace_buffer[0][0] > self.hit_trace_window_size:
            self.hit_trace_buffer.popleft()
    
    def get_hit_ratio(self):
        hit_tokens = sum(x[1] for x in self.hit_trace_buffer)
        total_prompt_len = sum(x[2] for x in self.hit_trace_buffer)
        return hit_tokens / total_prompt_len if total_prompt_len > 0 else 0
            
    def waiting_queue_prefix_hit(self, s: SchedulingMetricsReqInput):
        max_pref_length = 0
        for req in self.forward_queue:
            for i, (a, b) in enumerate(zip(s.input_ids, req.input_ids)):
                if a != b:
                    max_pref_length = max(max_pref_length, i)
                    break
            else:
                if len(req.input_ids) >= len(s.input_ids):
                    max_pref_length = len(s.input_ids) - 1
                    break
        return max_pref_length

    def exposed_scheduler_metrics_request(self, recv_req: SchedulingMetricsReqInput):
        """
        Performs a prefix match on the data and collect metrics that could be useful for a global load balancer.

        Note: Handle as a seperate async request to avoid blocking the existing function
        """
        start_time = time.time()
        prefix_indices, last_node = self.tree_cache.match_prefix(recv_req.input_ids)
        # max_prefix_match = max(len(prefix_indices), self.waiting_queue_prefix_hit(recv_req))
        max_prefix_match = len(prefix_indices)
        match_overhead = time.time() - start_time
        average_waiting_queue_len = sum(self.forward_queue_len_buffer) / len(self.forward_queue_len_buffer) if len(self.forward_queue_len_buffer) > 0 else 0
        average_running_batch_len = sum(self.running_batch_len_buffer) / len(self.running_batch_len_buffer) if len(self.running_batch_len_buffer) > 0 else 0     

        out = SchedulingMetricsOut(
            rid=recv_req.rid,
            input_len=len(recv_req.input_ids),
            waiting_queue_len=average_waiting_queue_len,
            running_req_len=average_running_batch_len,
            prefix_match_len= max_prefix_match,
            token_kv_available_size=self.token_to_kv_pool.available_size(),
            evicatable_size=self.tree_cache.evictable_size(),
            tree_cache_metrics_hit=self.tree_cache_metrics["hit"],
            tree_cache_metrics_total=self.tree_cache_metrics["total"],
            total_radix_cache_processing_time=time.time() - start_time,
            queue_processing_time=time.time(),
            inner_router_time=0,
            waiting_time_tokenizer_manager=0,
            matching_overhead=match_overhead * 1000,
            manager_dispatch_time=0,
            manager_recv_time=0,
        )
        return out

    def exposed_get_migration_candidates(self):
        if self.tp_size != 1:
            raise ValueError("TP>1 migration is not considered when implemented")
        ret = self.forward_queue
        self.forward_queue = []
        return ret

    def exposed_step(self, recv_reqs):
        if self.tp_size != 1:
            recv_reqs = obtain(recv_reqs)

        try:
            # Recv requests
            for recv_req in recv_reqs:
                if isinstance(recv_req, TokenizedGenerateReqInput):
                    self.handle_generate_request(recv_req)
                elif isinstance(recv_req, FlushCacheReq):
                    self.flush_cache()
                else:
                    raise ValueError(f"Invalid request: {recv_req}")

            # Forward
            if self.chunk_prefill_budget > 1:
                self.budget_forward_step()
            else:
                self.forward_step()
        except Exception:
            logger.error("Exception in ModelRpcClient:\n" + get_exception_traceback())

        # Return results
        ret = self.out_pyobjs
        self.out_pyobjs = []
        return ret
    
    def _schedule_running(
        self, 
        budget: SchedulingBudget
    ) -> Tuple[List[Req], Batch]:
        batch = self.running_batch
        if batch is None or batch.is_empty():
            return [], None
        current_running_idx = sorted(
            [i for i in range(len(batch.reqs))],
            key=lambda i: (batch.reqs[i].arrival_time, len(batch.reqs[i].output_ids))
        )
        req_pool_indices_cpu = batch.req_pool_indices.cpu().tolist()
        preempted = []
        scheduled = []
        batch.out_cache_loc = None
        out_cache_locs = []
        while current_running_idx:
            if budget.get_remaining_token_budget() <= 0:
                break
            # try to schedule each request
            target_idx = current_running_idx.pop(0)
            target_to_schedule = batch.reqs[target_idx]
            # 1. check required new memory and evict if needed
            # logger.debug(f'output_len: {len(target_to_schedule.output_ids)}, unfinished: {target_to_schedule.get_num_unfinished_tokens()}, budget: {budget.get_remaining_token_budget()}')
            new_tokens = min(target_to_schedule.get_num_unfinished_tokens(), budget.get_remaining_token_budget())
            if (batch.token_to_kv_pool.available_size() < new_tokens 
                and not batch.tree_cache.disable):
                batch.tree_cache.evict(new_tokens, batch.token_to_kv_pool.free)
            # 2. if not enough, evict running requests
            while batch.token_to_kv_pool.available_size() < new_tokens:
                if not current_running_idx:
                    evict_idx = target_idx
                    evict_req = target_to_schedule
                else:
                    evict_idx = current_running_idx.pop()
                    evict_req = batch.reqs[evict_idx]
                    
                batch.tree_cache.dec_ref_counter(evict_req.last_node)
                token_indices = batch.req_to_token_pool.req_to_token[
                    req_pool_indices_cpu[evict_idx]
                ][: evict_req.num_cached_tokens]
                batch.token_to_kv_pool.free(token_indices)
                batch.req_to_token_pool.free(req_pool_indices_cpu[evict_idx])
                self.recomputed_tokens += len(evict_req.input_ids)
                
                evict_req.reset_state()
                preempted.append(evict_req)
                # not enought memory to schedule
                if evict_idx == target_idx:
                    break
            else:
            #3. schedule it, allocate and set workload
                budget.schedule_new_tokens(new_tokens)
                scheduled.append(target_idx)
                out_cache_loc = batch.token_to_kv_pool.alloc(new_tokens)
                batch.req_to_token_pool.req_to_token[
                    req_pool_indices_cpu[target_idx],
                    target_to_schedule.num_cached_tokens : target_to_schedule.num_cached_tokens + new_tokens
                ] = out_cache_loc
                target_to_schedule.num_inflight_tokens = new_tokens
                out_cache_locs.append(out_cache_loc)
        
        if current_running_idx:
            delayed_batch = batch.copy_from(current_running_idx)
        else:
            delayed_batch = None
               
        if len(scheduled) < len(batch.reqs):
            logger.debug(f'GPU: {self.current_gpu} preempted/delayed {len(batch.reqs) - len(scheduled)} requests')
            batch.filter_batch(scheduled)
            
        # set out_cache_loc
        if not out_cache_locs:
            pass
        out_cache_locs = torch.cat(out_cache_locs, dim=0)
        batch.out_cache_loc = out_cache_locs
        batch.out_cache_cont_start = batch.out_cache_cont_end = None
        
        batch.prepare_for_decode_v2()
        num_batched_tokens = batch.input_ids.shape[0]
        num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
        unique_kvs = self.tree_cache.total_unique_kv_tokens(batch.reqs)
        detail_batch_logger(
            f"GPU: {self.current_gpu} "
            f"schedule running: "
            f"batch.num_reqs: {len(batch.reqs)}, "
            f"input ids: {num_batched_tokens}, "
            f"attention tokens: {num_attention_tokens}, "
            f"unique kv tokens: {unique_kvs}"
        )
        return preempted, delayed_batch

    # TODO: add log prob
    @torch.inference_mode()
    def budget_forward_step(self, forward_simulation=None, current_time=None):
        if current_time is None:
            self.current_time = time.time()
        else:
            self.current_time = current_time
        # For fast populate
        if self.token_to_kv_pool.available_size() / self.max_total_num_token >= 0.5:
            return self.forward_step(forward_simulation, current_time)
        start_scheduling = time.time()
        budget = SchedulingBudget(self.chunk_prefill_budget, 0)
        if self.delayed_batch:
            if self.running_batch:
                self.running_batch.concat(self.delayed_batch)
            else:
                self.running_batch = self.delayed_batch
            self.delayed_batch = None
        preempted, delayed_batch = self._schedule_running(budget)
        self.delayed_batch = delayed_batch
        self.forward_queue.extend(preempted)
        if not self.disable_regex_jump_forward and self.running_batch:
            # check for jump-forward
            jump_forward_reqs = self.running_batch.check_for_jump_forward()

            # check for image jump-forward
            for req in jump_forward_reqs:
                if req.pixel_values is not None:
                    (
                        req.input_ids,
                        req.image_offset,
                    ) = self.model_runner.model.pad_input_ids(
                        req.input_ids,
                        req.pad_value,
                        req.pixel_values.shape,
                        req.image_size,
                    )
            self.forward_queue.extend(jump_forward_reqs)
            
        if self.running_batch and not self.running_batch.is_empty():
            self.decode_forward_ct = (self.decode_forward_ct + 1) % (1 << 30)
            
        if budget.get_remaining_token_budget() <= 0:
            scheduled_waiting_batch = None
        else:
            schedule_waiting_start = time.time()
            # NOTE: this add too much overhead, only recheck prefix hit every 10 iterations
            # self.iter_cnt = (self.iter_cnt + 1) % 5
            # self.check_req_hit(self.forward_queue, self.iter_cnt != 0)
            self.check_req_hit(self.forward_queue, False)
            if self.schedule_heuristic == 'fcfs-mpq':
                while self.forward_queue:
                    req = self.forward_queue.pop()
                    # Assign to the corresponding priority queue based on hit ratio
                    hit_ratio = len(req.prefix_indices) / len(req.input_ids)
                    if hit_ratio == 0:
                        group = 0
                    else:
                        group = math.ceil(hit_ratio * 10) - 1
                    self.multi_priority_queue[group].append(req)
                schedule_group_idx = self.get_schedule_group()
                if schedule_group_idx is None:
                    scheduled_waiting_batch = None
                else:
                    # tune k to trade off between cache hit and fairness
                    k = 10
                    max_schedule_allowed = (schedule_group_idx + 1) * k

                    # Get priority queue
                    target_waiting_queue = self.scheduler.get_priority_queue(self.multi_priority_queue[schedule_group_idx])
                    self.check_req_hit(target_waiting_queue)
                    scheduled_waiting_batch = self.schedule_within_group(target_waiting_queue, max_schedule_allowed, budget)
                if scheduled_waiting_batch is not None:
                    self.multi_priority_queue[schedule_group_idx] = [x for x in target_waiting_queue if x not in scheduled_waiting_batch.reqs]
            else:
                self.forward_queue = self.scheduler.get_priority_queue(self.forward_queue)
                # scheduled_waiting_batch = self._schedule_waiting(budget)
                scheduled_waiting_batch = self.schedule_within_group(self.forward_queue, self.max_num_running_seq, budget)
                if scheduled_waiting_batch is not None:
                    self.forward_queue = [x for x in self.forward_queue if x not in scheduled_waiting_batch.reqs]
            self.schedule_waiting_overhead += time.time() - schedule_waiting_start
        
        if scheduled_waiting_batch is not None:
            scheduled_waiting_batch.prepare_for_extend_v2(self.model_config.vocab_size, self.int_token_logit_bias)
            if self.running_batch:
                self.running_batch.concat(scheduled_waiting_batch)
            else:
                self.running_batch = scheduled_waiting_batch
        self.total_scheduling_overhead += time.time() - start_scheduling
        if not self.running_batch or self.running_batch.is_empty():
            return []
        # Logging
        batch = self.running_batch
        num_batched_tokens = batch.input_ids.shape[0]
        num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
        unique_kvs = self.tree_cache.total_unique_kv_tokens(batch.reqs)
        if self.tp_rank == 0:
            logger.debug(
                f"GPU: {self.current_gpu} "
                f"batch.num_reqs: {len(batch.reqs)}, "
                f"batch.inflight_tokens: {num_batched_tokens}, "
                f"attention tokens: {num_attention_tokens}, "
                f"unique kv tokens: {unique_kvs}, "
                f"#remaining_req: {self.num_waiting_reqs()}, "
                f"free_gpu_mem: {self.token_to_kv_pool.available_size() / self.max_total_num_token:.2f}. "
                f"windowed_cache_hit_rate: {100.0 * self.get_hit_ratio():.2f}%. "
            )
        self.total_forwarded_tokens += num_batched_tokens
        self.total_cache_hit_tokens += sum(
            len(req.prefix_indices) if req.prefix_indices is not None else 0
            for req in batch.reqs
        )
        
        # self.running_batch.prepare_for_isolate_extend_decode()
        forward_time = 0
        if num_batched_tokens > 0:
            # Forward
            if forward_simulation is None:
                # logger.debug(self.running_batch)
                s = time.time()
                if not self.use_sleep_forwarding:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    logits, (_, _, last_logprobs) = self.model_runner.forward(
                        batch, ForwardMode.EXTEND
                    )
                    end_event.record()
                    next_token_ids, _ = batch.sample(logits)
                else:
                    vocab_size = self.model_config.vocab_size
                    logits = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
                    next_token_ids = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
                    time.sleep(self.gpu_config.forward_simulation[0](
                        len(batch.reqs),
                        num_batched_tokens,
                        num_attention_tokens,
                        batch.input_id_lengths,
                        unique_kvs,
                        batch.seq_lens,
                    ))
                    _ = batch.sample(logits)
                    last_logprobs = None
                forward_time = time.time() - s
                forward_time = forward_time
            else:
                vocab_size = self.model_config.vocab_size
                logits = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
                next_token_ids = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
                forward_time = forward_simulation[0](
                    len(batch.reqs),
                    num_batched_tokens,
                    num_attention_tokens,
                    batch.input_id_lengths,
                    unique_kvs,
                    batch.seq_lens,
                )
                _ = batch.sample(logits)
                last_logprobs = None
            next_token_ids = next_token_ids.cpu().tolist()
        else:
            next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)
            logits = last_logprobs = None
            start_event, end_event = None, None
            
        # Only batch transfer the selected logprobs of the next token to CPU to reduce overhead.
        reqs = batch.reqs
        if last_logprobs is not None:
            last_logprobs = last_logprobs[
                torch.arange(len(reqs)), next_token_ids
            ].tolist()

        # Check finish condition
        for i, (req, next_tok_id) in enumerate(zip(reqs, next_token_ids)):
            req.update_after_step()
            if req.get_num_unfinished_tokens() == 0:
                req.completion_tokens_wo_jump_forward += 1
                req.output_ids.append(next_tok_id)
                req.check_finished()

                if last_logprobs is not None:
                    req.token_logprob.append((next_tok_id, last_logprobs[i]))

        self.handle_finished_requests(batch)
        if batch.is_empty():
            self.running_batch = None
        
        model_forward_time = forward_time * 1000
        if not self.use_sleep_forwarding and forward_simulation is None:
            if start_event and end_event:
                end_event.synchronize()
                model_forward_time = start_event.elapsed_time(end_event)
        if model_forward_time > 0:
            detail_batch_logger(
                f'GPU: {self.current_gpu} '
                f"forward time: {model_forward_time:.2f} ms"
            )
        return [forward_time]
        

    @torch.inference_mode()
    def forward_step(self, forward_simulation=None, current_time=None):
        if current_time is None:
            self.current_time = time.time()
        else:
            self.current_time = current_time
        if self.schedule_heuristic == 'fcfs-mpq':
            new_batch = self.get_new_fill_batch_v2()
        else:
            new_batch = self.get_new_fill_batch()
        forward_times = []
        if new_batch is not None:
            # Run new fill batch
            forward_times.append(self.forward_fill_batch(new_batch, forward_simulation))

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
        else:
            # Run decode batch
            if self.running_batch is not None:
                # Run a few decode batches continuously for reducing overhead
                for _ in range(10):
                    forward_time = self.forward_decode_batch(self.running_batch, forward_simulation)
                    forward_times.append(forward_time)
                    if self.running_batch.is_empty():
                        self.running_batch = None
                        break

                    if self.out_pyobjs and self.running_batch.reqs[0].stream:
                        break

                    if self.running_batch is not None and self.tp_rank == 0:
                        if self.decode_forward_ct % 40 == 0:
                            num_used = self.max_total_num_token - (
                                self.token_to_kv_pool.available_size()
                                + self.tree_cache.evictable_size()
                            )
                            logger.info(
                                f"#running-req: {len(self.running_batch.reqs)}, "
                                f"#token: {num_used}, "
                                f"token usage: {num_used / self.max_total_num_token:.2f}, "
                                f"#queue-req: {len(self.forward_queue)}"
                            )
            else:
                # check the available size
                available_size = (
                    self.token_to_kv_pool.available_size()
                    + self.tree_cache.evictable_size()
                )
                if available_size != self.max_total_num_token:
                    warnings.warn(
                        "Warning: "
                        f"available_size={available_size}, max_total_num_token={self.max_total_num_token}\n"
                        "KV cache pool leak detected!"
                    )
        total_forward_time = 0
        if (self.use_sleep_forwarding or forward_simulation) and forward_times:
            total_forward_time = sum(forward_times) * 1000
        elif forward_times:
            for start, end in forward_times:
                if start and end:
                    end.synchronize()
                    total_forward_time += start.elapsed_time(end)
        if total_forward_time > 0:
            detail_batch_logger(
                f'GPU: {self.current_gpu} '
                f"forward time: {total_forward_time:.2f} ms"
            )

        return forward_times
    
    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        req = Req(recv_req.rid, recv_req.input_text, recv_req.input_ids, 
                  recv_req.arrival_time, recv_req.append_to_queue_time)
        req.pixel_values = recv_req.pixel_values
        if req.pixel_values is not None:
            req.pad_value = [
                (recv_req.image_hash) % self.model_config.vocab_size,
                (recv_req.image_hash >> 16) % self.model_config.vocab_size,
                (recv_req.image_hash >> 32) % self.model_config.vocab_size,
                (recv_req.image_hash >> 64) % self.model_config.vocab_size,
            ]
            req.image_size = recv_req.image_size
            req.input_ids, req.image_offset = self.model_runner.model.pad_input_ids(
                req.input_ids, req.pad_value, req.pixel_values.shape, req.image_size
            )
        req.sampling_params = recv_req.sampling_params
        req.return_logprob = recv_req.return_logprob
        req.logprob_start_len = recv_req.logprob_start_len
        req.stream = recv_req.stream
        req.tokenizer = self.tokenizer

        # Init regex fsm
        if req.sampling_params.regex is not None:
            req.regex_fsm = self.regex_fsm_cache.query(req.sampling_params.regex)
            if not self.disable_regex_jump_forward:
                req.jump_forward_map = self.jump_forward_cache.query(
                    req.sampling_params.regex
                )

        # Truncate long prompts
        req.input_ids = req.input_ids[: self.model_config.context_len - 1]
        req.sampling_params.max_new_tokens = min(
            req.sampling_params.max_new_tokens,
            self.model_config.context_len - 1 - len(req.input_ids),
            self.max_total_num_token - 128 - len(req.input_ids),
        )
        self.forward_queue.append(req)
    
    def check_req_hit(self, queue: List[Req], skip: bool=False):
        if not skip:
            for req in queue:
                prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
                if req.return_logprob:
                    prefix_indices = prefix_indices[: req.logprob_start_len]
                req.extend_input_len = len(req.input_ids) - len(prefix_indices)
                req.prefix_indices = prefix_indices
                req.last_node = last_node
        else:
            for req in queue:
                req.extend_input_len = len(req.input_ids)
                req.prefix_indices = []
                req.last_node = self.tree_cache.root_node
    
    def get_schedule_group(self):
        start_from = None
        while not self.multi_priority_queue[self.start_schedule_from]:
            if start_from is None:
                start_from = self.start_schedule_from
            elif start_from == self.start_schedule_from:
                return None
            self.start_schedule_from = (self.start_schedule_from + 9) % 10
        else:
            start_from = self.start_schedule_from
            self.start_schedule_from = (self.start_schedule_from + 9) % 10
            return start_from
    
    def schedule_within_group(
        self, 
        target_waiting_queue: List[Req], 
        max_schedule_allowed: int,
        budget: SchedulingBudget = None
    ):
        # Add requests if there is available space
        can_run_list = []
        new_batch_total_tokens = 0
        new_batch_input_tokens = 0
        total_batched_prompt_len = 0

        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        if self.running_batch:
            available_size -= sum(
                [
                    (r.max_new_tokens() - len(r.output_ids)) * self.new_token_ratio
                    for r in self.running_batch.reqs
                ]
            )
        req: Req
        for req in target_waiting_queue:
            if budget and budget.get_remaining_token_budget() <= 0:
                break
            if max_schedule_allowed <= 0:
                break
            if req.return_logprob:
                # Need at least two tokens to compute normalized logprob
                if req.extend_input_len < 2:
                    delta = 2 - req.extend_input_len
                    req.extend_input_len += delta
                    req.prefix_indices = req.prefix_indices[:-delta]
                    if req.image_offset is not None:
                        req.image_offset += delta
            if req.extend_input_len == 0 and req.max_new_tokens() > 0:
                # Need at least one token to compute logits
                req.extend_input_len = 1
                req.prefix_indices = req.prefix_indices[:-1]
                if req.image_offset is not None:
                    req.image_offset += 1
            if budget:
                num_new_tokens = min(budget.get_remaining_token_budget(), req.extend_input_len)
            else:
                num_new_tokens = req.extend_input_len
            if (
                req.extend_input_len + req.max_new_tokens() + new_batch_total_tokens
                < available_size
                and req.extend_input_len + new_batch_input_tokens
                < self.max_prefill_num_token
            ):
                delta = self.tree_cache.inc_ref_counter(req.last_node)
                available_size += delta

                if not (
                    req.extend_input_len + req.max_new_tokens() + new_batch_total_tokens
                    < available_size
                ):
                    # Undo the insertion
                    delta = self.tree_cache.dec_ref_counter(req.last_node)
                    available_size += delta
                else:
                    req.num_inflight_tokens = num_new_tokens
                    req.num_cached_tokens = len(req.prefix_indices)
                    if budget:
                        budget.schedule_new_tokens(num_new_tokens)
                    # Add this request to the running batch
                    self.token_to_kv_pool.add_refs(req.prefix_indices)
                    can_run_list.append(req)
                    new_batch_total_tokens += (
                        req.extend_input_len + req.max_new_tokens()
                    )
                    new_batch_input_tokens += req.num_inflight_tokens
                    total_batched_prompt_len += len(req.input_ids)
                    max_schedule_allowed -= 1
            elif self.schedule_heuristic == "fcfs-s":
                    break
            elif self.schedule_heuristic == 'fcfs-escape':
                if self.current_time - req.arrival_time >= 10:
                    break
                    
        if len(can_run_list) == 0:
            return None

        if self.tp_rank == 0:
            running_req = (
                0 if self.running_batch is None else len(self.running_batch.reqs)
            )
            hit_tokens = sum(len(x.prefix_indices) for x in can_run_list)
            self.tree_cache_metrics["total"] += (
                total_batched_prompt_len
            ) / 10**9
            self.tree_cache_metrics["hit"] += hit_tokens / 10**9
            tree_cache_hit_rate = (
                self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"]
            )
            self.update_hit_trace(self.current_time, hit_tokens, total_batched_prompt_len)
            logger.info(
                f"GPU: {self.current_gpu} "
                f"new fill batch. #seq: {len(can_run_list)}. "
                f"#cached_token: {hit_tokens}. "
                f"#new_token: {new_batch_input_tokens}. "
                f"#remaining_req: {self.num_waiting_reqs() - len(can_run_list)}. "
                f"#running_req: {running_req}. "
                f"tree_cache_hit_rate: {100.0 * tree_cache_hit_rate:.2f}%. "
                f"windowed_cache_hit_rate: {100.0 * self.get_hit_ratio():.2f}%. "
                f"hit_tokens: {hit_tokens}. "
                f"free_gpu_mem: {self.token_to_kv_pool.available_size() / self.max_total_num_token:.2f}. "
            )
            # logger.debug(
            #     f"fsm_cache_hit_rate: {100.0 * self.regex_fsm_cache.get_cache_hit_rate():.2f}%. "
            #     f"fsm_cache_avg_init_time: {self.regex_fsm_cache.get_avg_init_time():.2f}s. "
            #     f"ff_cache_hit_rate: {100.0 * self.jump_forward_cache.get_cache_hit_rate():.2f}%. "
            #     f"ff_cache_avg_init_time: {self.jump_forward_cache.get_avg_init_time():.2f}s. "
            #     f"hit_tokens: {hit_tokens}."
            # )
            self.forward_queue_len_buffer.append(self.num_waiting_reqs())
            self.running_batch_len_buffer.append(running_req)

        new_batch = Batch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
        )
        return new_batch
    
    def get_new_fill_batch_v2(self):
        if (
            self.running_batch is not None
            and len(self.running_batch.reqs) > self.max_num_running_seq
        ):
            return None
        schedule_waiting_start = time.time()
                
        self.check_req_hit(self.forward_queue)
        while self.forward_queue:
            req = self.forward_queue.pop()
            # Assign to the corresponding priority queue based on hit ratio
            hit_ratio = len(req.prefix_indices) / len(req.input_ids)
            if hit_ratio == 0:
                group = 0
            else:
                group = math.ceil(hit_ratio * 10) - 1
            self.multi_priority_queue[group].append(req)
        
        schedule_group_idx = self.get_schedule_group()
        if schedule_group_idx is None:
            self.total_scheduling_overhead += time.time() - schedule_waiting_start
            self.schedule_waiting_overhead += time.time() - schedule_waiting_start
            return None
        # tune k to trade off between cache hit and fairness
        k = 5
        max_schedule_allowed = (schedule_group_idx + 1) * k

        # Get priority queue
        target_waiting_queue = self.scheduler.get_priority_queue(self.multi_priority_queue[schedule_group_idx])
        # NOTE: this is necessary
        # 1. hit ratio will shift over time
        # 2. if you set longer prefix indices the result is incorrect
        self.check_req_hit(target_waiting_queue)
        new_batch = self.schedule_within_group(target_waiting_queue, max_schedule_allowed)
        if new_batch is None:
            self.total_scheduling_overhead += time.time() - schedule_waiting_start
            self.schedule_waiting_overhead += time.time() - schedule_waiting_start
            return None
        
        self.multi_priority_queue[schedule_group_idx] = [x for x in target_waiting_queue if x not in new_batch.reqs]
        if self.log_prefix_hit:
            self.prefix_hit_trace.append({x.rid: [x.input_text[:20], len(x.prefix_indices)] for x in new_batch.reqs})
        self.schedule_waiting_overhead += time.time() - schedule_waiting_start
        self.total_scheduling_overhead += time.time() - schedule_waiting_start
        return new_batch
        

    def get_new_fill_batch(self):
        if (
            self.running_batch is not None
            and len(self.running_batch.reqs) > self.max_num_running_seq
        ):
            return None
        schedule_waiting_start = time.time()

        self.check_req_hit(self.forward_queue)
        # Get priority queue
        self.forward_queue = self.scheduler.get_priority_queue(self.forward_queue)

        # Add requests if there is available space
        new_batch = self.schedule_within_group(self.forward_queue, self.max_num_running_seq)
        if new_batch is None:
            self.total_scheduling_overhead += time.time() - schedule_waiting_start
            self.schedule_waiting_overhead += time.time() - schedule_waiting_start
            return None
        self.forward_queue = [x for x in self.forward_queue if x not in new_batch.reqs]
        if self.log_prefix_hit:
            self.prefix_hit_trace.append({x.rid: [x.input_text[:20], len(x.prefix_indices)] for x in new_batch.reqs})
        self.schedule_waiting_overhead += time.time() - schedule_waiting_start
        self.total_scheduling_overhead += time.time() - schedule_waiting_start
        return new_batch
    
    def dump_prefix_hit_trace(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.prefix_hit_trace, f)        

    def forward_fill_batch(self, batch: Batch, forward_simulation=None):
        # Build batch tensors
        batch.prepare_for_extend_v2(
            self.model_config.vocab_size, self.int_token_logit_bias
        )

        logprobs = None
        forward_time = 0
        num_batched_tokens = batch.input_ids.shape[0]
        num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
        unique_kvs = self.tree_cache.total_unique_kv_tokens(batch.reqs)
        self.total_forwarded_tokens += num_batched_tokens
        self.total_cache_hit_tokens += sum(
            len(req.prefix_indices) if req.prefix_indices is not None else 0
            for req in batch.reqs
        )
        # if self.tp_rank == 0:
        #     logging.info(
        #         f"GPU: {self.current_gpu} "
        #         f"batch.extend_num_tokens: {batch.extend_num_tokens}, "
        #         f"num reqs: {len(batch.reqs)}, "
        #         f"input ids: {num_batched_tokens}, "
        #         f"attention tokens: {num_attention_tokens}, "
        #         f"tree unique ref nodes : {unique_kvs}"
        #         # f"prefix indices: {batch.prefix_lens}"
        #     )

        if batch.extend_num_tokens != 0:
            if forward_simulation is None:
                # logger.debug(batch)
                # Forward
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                s = time.time()
                if not self.use_sleep_forwarding:
                    logits, (
                        prefill_logprobs,
                        normalized_logprobs,
                        last_logprobs,
                    ) = self.model_runner.forward(batch, ForwardMode.EXTEND)
                    if prefill_logprobs is not None:
                        logprobs = prefill_logprobs.cpu().tolist()
                        normalized_logprobs = normalized_logprobs.cpu().tolist()
                    next_token_ids, _ = batch.sample(logits)
                else:
                    vocab_size = self.model_config.vocab_size
                    logits = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
                    next_token_ids = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
                    time.sleep(self.gpu_config.forward_simulation[0](
                        len(batch.reqs),
                        num_batched_tokens,
                        num_attention_tokens,
                        batch.input_id_lengths,
                        unique_kvs,
                        batch.seq_lens,
                    ))
                    _ = batch.sample(logits)
                    logprobs = normalized_logprobs = last_logprobs = None
                end_event.record()
                forward_time = time.time() - s
                forward_time = forward_time
            else:
                vocab_size = self.model_config.vocab_size
                logits = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
                next_token_ids = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
                forward_time = forward_simulation[0](
                    len(batch.reqs),
                    num_batched_tokens,
                    num_attention_tokens,
                    batch.input_id_lengths,
                    unique_kvs,
                    batch.seq_lens,
                )
                _ = batch.sample(logits)
                logprobs = normalized_logprobs = last_logprobs = None
            next_token_ids = next_token_ids.cpu().tolist()
        else:
            next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)
            logits = logprobs = normalized_logprobs = last_logprobs = None
            start_event, end_event = None, None

        # Only batch transfer the selected logprobs of the next token to CPU to reduce overhead.
        reqs = batch.reqs
        if last_logprobs is not None:
            last_logprobs = (
                last_logprobs[torch.arange(len(reqs)), next_token_ids].cpu().tolist()
            )

        # Check finish condition
        pt = 0
        for i, req in enumerate(reqs):
            req.update_after_step()
            if req.get_num_unfinished_tokens() == 0:
                req.completion_tokens_wo_jump_forward += 1
                req.output_ids = [next_token_ids[i]]
                req.check_finished()

                if logprobs is not None:
                    req.logprob = logprobs[pt : pt + req.extend_input_len - 1]
                    req.normalized_logprob = normalized_logprobs[i]

                    # If logprob_start_len > 0, then first logprob_start_len prompt tokens
                    # will be ignored.
                    prompt_token_len = len(req.logprob)
                    token_ids = req.input_ids[-prompt_token_len:] + [next_token_ids[i]]
                    token_logprobs = req.logprob + [last_logprobs[i]]
                    req.token_logprob = list(zip(token_ids, token_logprobs))
                    if req.logprob_start_len == 0:
                        req.token_logprob = [(req.input_ids[0], None)] + req.token_logprob
                    pt += req.extend_input_len

        self.handle_finished_requests(batch)
        if self.use_sleep_forwarding:
            return forward_time
        if forward_simulation is None:
            return start_event, end_event
        return forward_time
    
    def test_extend_decode(self, batch: Batch, forward_simulation=None):
        budget = SchedulingBudget(self.max_num_running_seq, 0)
        preempted, delayed_batch = self._schedule_running(budget)
        assert delayed_batch is None
        self.forward_queue.extend(preempted)
        
        num_batched_tokens = batch.input_ids.shape[0]
        num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
        unique_kvs = self.tree_cache.total_unique_kv_tokens(batch.reqs)
        if self.tp_rank == 0:
            detail_batch_logger(
                f"GPU: {self.current_gpu} "
                f"batch.num_reqs: {len(batch.reqs)}, "
                f"input ids: {num_batched_tokens}, "
                f"attention tokens: {num_attention_tokens}, "
                f"unique kv tokens: {unique_kvs}"
            )
        forward_time = 0
        # Forward
        if forward_simulation is None:
            # logger.debug(batch)
            s = time.time()
            if not self.use_sleep_forwarding:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()        
                logits, (_, _, last_logprobs) = self.model_runner.forward(
                    batch, ForwardMode.EXTEND
                )
                end_event.record()
                next_token_ids, _ = batch.sample(logits)
            else:
                vocab_size = self.model_config.vocab_size
                logits = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
                next_token_ids = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
                time.sleep(self.gpu_config.forward_simulation[1](
                    len(batch.reqs),
                    num_batched_tokens,
                    num_attention_tokens,
                    unique_kvs
                ))   
                _ = batch.sample(logits)
                last_logprobs = None
            forward_time = time.time() - s
            forward_time = forward_time
        else:
            vocab_size = self.model_config.vocab_size
            logits = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
            next_token_ids = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
            forward_time = forward_simulation[1](
                len(batch.reqs),
                num_batched_tokens,
                num_attention_tokens,
                unique_kvs
            )
            _ = batch.sample(logits)
            last_logprobs = None
        next_token_ids = next_token_ids.cpu().tolist()

        # Only batch transfer the selected logprobs of the next token to CPU to reduce overhead.
        reqs = batch.reqs
        if last_logprobs is not None:
            last_logprobs = last_logprobs[
                torch.arange(len(reqs)), next_token_ids
            ].tolist()

        # Check finish condition
        for i, (req, next_tok_id) in enumerate(zip(reqs, next_token_ids)):
            req.update_after_step()
            if req.get_num_unfinished_tokens() == 0:
                req.completion_tokens_wo_jump_forward += 1
                req.output_ids.append(next_tok_id)
                req.check_finished()

                if last_logprobs is not None:
                    req.token_logprob.append((next_tok_id, last_logprobs[i]))

        self.handle_finished_requests(batch)
        if self.use_sleep_forwarding:
            return forward_time
        if forward_simulation is None:
            return start_event, end_event
        return forward_time
    
    def forward_decode_batch(self, batch: Batch, forward_simulation=None):
        start_decoding_schedule = time.time()
        # check if decode out of memory
        if not batch.check_decode_mem():
            old_ratio = self.new_token_ratio
            self.new_token_ratio = min(old_ratio + self.new_token_ratio_step[1], 1.0)

            retracted_reqs = batch.retract_decode()
            self.recomputed_tokens += sum(len(r.input_ids) for r in retracted_reqs)
            logger.info(
                f"GPU {self.current_gpu}: decode out of memory happened, "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self.forward_queue.extend(retracted_reqs)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_step[0],
                self.min_new_token_ratio,
            )

        if not self.disable_regex_jump_forward:
            # check for jump-forward
            jump_forward_reqs = batch.check_for_jump_forward()

            # check for image jump-forward
            for req in jump_forward_reqs:
                if req.pixel_values is not None:
                    (
                        req.input_ids,
                        req.image_offset,
                    ) = self.model_runner.model.pad_input_ids(
                        req.input_ids,
                        req.pad_value,
                        req.pixel_values.shape,
                        req.image_size,
                    )

            self.forward_queue.extend(jump_forward_reqs)
            if batch.is_empty():
                if self.use_sleep_forwarding:
                    return 0
                if forward_simulation is None:
                    return None, None
                return 0

        # Update batch tensors
        self.decode_forward_ct = (self.decode_forward_ct + 1) % (1 << 30)
        batch.prepare_for_decode()
        for req in batch.reqs:
            req.num_inflight_tokens = 1

        num_batched_tokens = batch.input_ids.shape[0]
        num_attention_tokens = batch.seq_lens.cpu().numpy().sum()
        unique_kvs = self.tree_cache.total_unique_kv_tokens(batch.reqs)
        if self.tp_rank == 0:
            detail_batch_logger(
                f"GPU: {self.current_gpu} "
                f"batch.num_reqs: {len(batch.reqs)}, "
                f"input ids: {num_batched_tokens}, "
                f"attention tokens: {num_attention_tokens}, "
                f"tree unique ref nodes : {unique_kvs}"
            )
        self.total_forwarded_tokens += num_batched_tokens
        self.total_scheduling_overhead += time.time() - start_decoding_schedule
        
        forward_time = 0
        # Forward
        if forward_simulation is None:
            s = time.time()
            # logger.debug(batch)
            if not self.use_sleep_forwarding:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()        
                logits, (_, _, last_logprobs) = self.model_runner.forward(
                    batch, ForwardMode.DECODE
                )
                end_event.record()
                next_token_ids, _ = batch.sample(logits)
            else:
                vocab_size = self.model_config.vocab_size
                logits = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
                next_token_ids = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
                time.sleep(self.gpu_config.forward_simulation[1](
                    len(batch.reqs),
                    num_batched_tokens,
                    num_attention_tokens,
                    unique_kvs
                ))   
                _ = batch.sample(logits)
                last_logprobs = None
            forward_time = time.time() - s
            forward_time = forward_time
        else:
            vocab_size = self.model_config.vocab_size
            logits = torch.ones((len(batch.reqs), vocab_size), dtype=torch.float16, device="cuda")
            next_token_ids = torch.ones((len(batch.reqs)), dtype=torch.int32, device="cuda")
            forward_time = forward_simulation[1](
                len(batch.reqs),
                num_batched_tokens,
                num_attention_tokens,
                unique_kvs
            )
            _ = batch.sample(logits)
            last_logprobs = None
        next_token_ids = next_token_ids.cpu().tolist()

        # Only batch transfer the selected logprobs of the next token to CPU to reduce overhead.
        reqs = batch.reqs
        if last_logprobs is not None:
            last_logprobs = last_logprobs[
                torch.arange(len(reqs)), next_token_ids
            ].tolist()

        # Check finish condition
        for i, (req, next_tok_id) in enumerate(zip(reqs, next_token_ids)):
            req.update_after_step()
            if req.get_num_unfinished_tokens() == 0:
                req.completion_tokens_wo_jump_forward += 1
                req.output_ids.append(next_tok_id)
                req.check_finished()

                if last_logprobs is not None:
                    req.token_logprob.append((next_tok_id, last_logprobs[i]))

        self.handle_finished_requests(batch)
        if self.use_sleep_forwarding:
            return forward_time
        if forward_simulation is None:
            return start_event, end_event
        return forward_time

    def handle_finished_requests(self, batch: Batch):
        output_rids = []
        output_tokens = []
        output_and_jump_forward_strs = []
        output_hit_stop_str = []
        output_skip_special_tokens = []
        output_meta_info = []
        output_finished = []
        finished_indices = []
        unfinished_indices = []
        for i, req in enumerate(batch.reqs):
            if req.finished:
                finished_indices.append(i)
            else:
                unfinished_indices.append(i)

            if req.finished or (
                (
                    req.stream
                    and req.output_ids
                    and (
                        self.decode_forward_ct % self.stream_interval == 0
                        or len(req.output_ids) == 1
                    )
                )
            ):
                output_rids.append(req.rid)
                output_tokens.append(req.output_ids)
                output_and_jump_forward_strs.append(req.output_and_jump_forward_str)
                output_hit_stop_str.append(req.hit_stop_str)
                output_skip_special_tokens.append(
                    req.sampling_params.skip_special_tokens
                )

                meta_info = {
                    "prompt_tokens": len(req.input_ids),
                    "completion_tokens": len(req.input_ids)
                    + len(req.output_ids)
                    - req.prompt_tokens,
                    "completion_tokens_wo_jump_forward": req.completion_tokens_wo_jump_forward,
                    "arrival_time": req.arrival_time,
                    "append_to_queue_time": req.append_to_queue_time,
                }
                if req.return_logprob:
                    meta_info["prompt_logprob"] = req.logprob
                    meta_info["token_logprob"] = req.token_logprob
                    meta_info["normalized_prompt_logprob"] = req.normalized_logprob
                output_meta_info.append(meta_info)
                output_finished.append(req.finished)

        # Send to detokenizer
        if output_rids:
            logger.debug(f'finished reqs: {len(output_rids)}')
            self.out_pyobjs.append(
                BatchTokenIDOut(
                    output_rids,
                    output_tokens,
                    output_and_jump_forward_strs,
                    output_hit_stop_str,
                    output_skip_special_tokens,
                    output_meta_info,
                    output_finished,
                )
            )

        # Remove finished reqs
        if finished_indices:
            # Update radix cache
            req_pool_indices_cpu = batch.req_pool_indices.cpu().tolist()
            for i in finished_indices:
                req = batch.reqs[i]
                req_pool_idx = req_pool_indices_cpu[i]
                token_ids = tuple(req.input_ids + req.output_ids)
                seq_len = len(token_ids) - 1
                indices = self.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
                prefix_len = self.tree_cache.insert(
                    token_ids[:seq_len], indices.clone()
                )

                self.token_to_kv_pool.free(indices[:prefix_len])
                self.req_to_token_pool.free(req_pool_idx)
                self.tree_cache.dec_ref_counter(req.last_node)

            # Update batch tensors
            if unfinished_indices:
                batch.filter_batch(unfinished_indices)
            else:
                batch.reqs = []


class ModelRpcService(rpyc.Service):
    exposed_ModelRpcServer = ModelRpcServer


class ModelRpcClient:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs, gpu_config: GPUConfig = None):
        tp_size = server_args.tp_size
        self.gpu_config = gpu_config

        if tp_size == 1:
            # Init model
            self.model_server = ModelRpcService().exposed_ModelRpcServer(
                0, server_args, port_args, False, gpu_config
            )

            # Wrap functions
            def async_wrap(f):
                async def _func(*args, **kwargs):
                    return f(*args, **kwargs)

                return _func

            self.step = async_wrap(self.model_server.exposed_step)
            self.push_req_step = async_wrap(self.model_server.handle_generate_request)
            self.get_migrate_candidates = async_wrap(self.model_server.exposed_get_migration_candidates)
            self.scheduler_metrics_request = async_wrap(
                self.model_server.exposed_scheduler_metrics_request
            )
            self.dump_prefix_hit_trace = async_wrap(self.model_server.dump_prefix_hit_trace)
        else:
            with ThreadPoolExecutor(tp_size) as executor:
                # Launch model processes
                rets = executor.map(start_model_process, port_args.model_rpc_ports)
                self.remote_services = [x[0] for x in rets]
                self.procs = [x[1] for x in rets]

                # Init model
                def init_model(i):
                    return self.remote_services[i].ModelRpcServer(
                        i, server_args, port_args
                    )

                self.model_servers = executor.map(init_model, range(tp_size))

            # Wrap functions
            def async_wrap(func_name):
                fs = [rpyc.async_(getattr(m, func_name)) for m in self.model_servers]

                async def _func(*args, **kwargs):
                    tasks = [f(*args, **kwargs) for f in fs]
                    await asyncio.gather(*[asyncio.to_thread(t.wait) for t in tasks])
                    return obtain(tasks[0].value)

                return _func

            self.step = async_wrap("step")
            # TODO: test push_req_step in TP mode
            self.push_req_step = async_wrap("handle_generate_request")
            self.scheduler_metrics_request = async_wrap(
                self.model_server.exposed_scheduler_metrics_request
            ) # TODO test metric collection in TP mode

def _init_service(port):
    t = ThreadedServer(
        ModelRpcService(),
        port=port,
        protocol_config={"allow_pickle": True, "sync_request_timeout": 1800},
    )
    t.start()


def start_model_process(port):
    proc = multiprocessing.Process(target=_init_service, args=(port,))
    proc.start()
    time.sleep(1)

    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect(
                "localhost",
                port,
                config={"allow_pickle": True, "sync_request_timeout": 1800},
            )
            break
        except ConnectionRefusedError:
            time.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise RuntimeError("init rpc env error!")

    assert proc.is_alive()
    return con.root, proc
