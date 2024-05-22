from collections import defaultdict
from datetime import datetime, timedelta
from benchmarks.benchmark_utils import RequestFuncOutput
from global_lru_cache import LPRadixCache, TreeNode
import time
import numpy as np
import threading
import heapq
from collections import deque
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
import logging
import copy
import math
from benchmarks.exp_configs.model_equations import LP_mistral_7b_A6000_sglang_extend_flashinfer as prefill_time
# from benchmarks.exp_configs.model_equations import LP_Llama3_70B_H100_sglang_extend_flashinfer as prefill_time
from ttft_overload_detector import TTFTWindowedOverloadedDetector

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

logger = logging.getLogger(__name__)

class SlidingWindowHistogram:
    def __init__(self, window_duration: timedelta, gpu_allocations, num_gpus=2, enable_miss_rate=True,avg_topt_per_gpu=None):
        self.window_duration = window_duration
        self.gpu_allocations = gpu_allocations
        self.histogram = defaultdict(int)
        self.node_to_count = defaultdict(int)

        self.timestamps: List[Tuple[datetime, TreeNode, TreeNode]] = []
        self.num_gpus = num_gpus
        self.enable_miss_rate = enable_miss_rate
        self.prev_mis_rates = {}
        self.hit_tokens = defaultdict(int)
        self.prompt_tokens = defaultdict(int)
        self.decoding_size = defaultdict(int)
        self.avg_topt_per_gpu = avg_topt_per_gpu

        self.current_prefill_cost_per_gpu = [0 for i in range(self.num_gpus)]
        self.current_decode_lengths_per_gpu = [0 for i in range(self.num_gpus)]
        self.per_node_prefill_cost = defaultdict(int)
        self.per_node_total_decode_lengths = defaultdict(int)

    @property
    def current_allocation_cost_per_gpu(self):
        costs = []
        for i in range(self.num_gpus):
            topt = np.median(self.avg_topt_per_gpu[i])
            costs.append(self.current_prefill_cost_per_gpu[i] + self.current_decode_lengths_per_gpu[i] * topt) 
        return costs

    def update(self, timestamp, node: TreeNode, leaf_node: TreeNode, runtime_idx, decoding_length):
        self.timestamps.append((timestamp, node, leaf_node))
        self.histogram[node] += 1 * leaf_node.context_length
        self.node_to_count[node] += 1
        self.decoding_size[node] = decoding_length
        self.per_node_total_decode_lengths[node] += decoding_length

        self.hit_tokens[node] += leaf_node.context_length - leaf_node.num_tokens
        self.prompt_tokens[node] += leaf_node.context_length

        self.prev_mis_rates[node] = 1 - (self.hit_tokens[node] / self.prompt_tokens[node])
        self._remove_old_entries(timestamp)

        self.update_prefill_cost_per_node(node, runtime_idx)
        self.current_decode_lengths_per_gpu[runtime_idx] += decoding_length

    def update_prefill_cost_per_node(self, node, runtime_idx):
        if runtime_idx not in self.gpu_allocations.get(node) or node not in self.prev_mis_rates or node not in self.node_to_count:
            return # only update gpu allocation for allocated nodes
        new_cost = self.get_prefill_cost(node)
        old_cost = self.per_node_prefill_cost[node]
        self.current_prefill_cost_per_gpu[runtime_idx] -= old_cost
        self.current_prefill_cost_per_gpu[runtime_idx] += new_cost
        self.per_node_prefill_cost[node] = new_cost

    def _remove_old_entries(self, current_timestamp):
        window_start = current_timestamp - self.window_duration
        while self.timestamps and self.timestamps[0][0] < window_start:
            timestamp, node, leaf_node = self.timestamps.pop(0)
            self.histogram[node] -= 1 * leaf_node.context_length
            self.node_to_count[node] -= 1
            self.hit_tokens[node] -= leaf_node.context_length - leaf_node.num_tokens
            self.prompt_tokens[node] -= leaf_node.context_length

            for gpu in range(self.num_gpus):
                self.update_prefill_cost_per_node(node, gpu)
                
            if self.histogram[node] <= 0:
                del self.histogram[node]
                del self.node_to_count[node]
                del self.prev_mis_rates[node]
                del self.hit_tokens[node]
                del self.prompt_tokens[node]
                del self.decoding_size[node]
                self.gpu_allocations[node] = set() # Reset the gpu allocation outside the time window

    def rename_node(self, old_node, new_node):
        if old_node in self.histogram:

            self.histogram[new_node] = self.histogram.pop(old_node)
            self.node_to_count[new_node] = self.node_to_count.pop(old_node)
            self.prev_mis_rates[new_node] = self.prev_mis_rates.pop(old_node)
            self.hit_tokens[new_node] = self.hit_tokens.pop(old_node)
            self.prompt_tokens[new_node] = self.prompt_tokens.pop(old_node)
            self.decoding_size[new_node] = self.decoding_size.pop(old_node)
            self.per_node_total_decode_lengths[new_node] = self.per_node_total_decode_lengths.pop(old_node)

            for gpu in range(self.num_gpus):
                self.per_node_prefill_cost[new_node] = self.per_node_prefill_cost[old_node]
                self.per_node_prefill_cost[old_node] = 0
                self.update_prefill_cost_per_node(new_node, gpu)

            timestamps = []
            for timestamp, important_node, leaf_node in self.timestamps:
                if important_node == old_node:
                    important_node = new_node
                timestamps.append((timestamp, important_node, leaf_node))
            self.timestamps = timestamps

    def query(self):
        return dict(self.histogram)

    def current_allocation_per_gpu(self):
        allocation = [0 for _ in range(self.num_gpus)]
        topts = []
        for i in range(self.num_gpus):
            topts.append(np.median(self.avg_topt_per_gpu[i]))
        node: TreeNode
        for node, cost in self.histogram.items():
            for gpu in self.gpu_allocations.get(node, {}):
                allocation[gpu] += self.get_node_cost(node, gpu, topts[gpu])
        return allocation

    def current_allocation_per_gpu_with_atleast_min_load(self, min_load=2):
        allocation = [0 for _ in range(self.num_gpus)]
        topts = []
        for i in range(self.num_gpus):
            topts.append(np.median(self.avg_topt_per_gpu[i]))

        node: TreeNode
        for node, cost in self.histogram.items():
            for gpu in self.gpu_allocations.get(node, {}):
                if self.node_to_count[node] < min_load:
                    continue
                allocation[gpu] += self.get_node_cost(node, gpu, topts[gpu])
        return allocation
    
    def get_prefill_cost(self, node: TreeNode):
        return self.prev_mis_rates[node] * self.node_to_count[node] * prefill_time(node.num_tokens, node.context_length) / len(self.gpu_allocations.get(node)) # potentionally divide by length of node.cached_gpus here
    
    def get_node_cost(self, node: TreeNode, gpu, tpot):
        prefill_cost = self.get_prefill_cost(node)
        output_len = self.decoding_size[node]
        if node.decode_length:
            output_len = np.median(node.decode_length)
        active_requests = node.ref_counter[gpu]
        decode_cost = active_requests * output_len * tpot
        return prefill_cost + decode_cost

    # FIXME: miss-rate is not per-gpu
    def get_eviction_prefill_cost(self, node: TreeNode, gpu, is_large_node: bool):
        if not is_large_node:
            return 0
        return self.prev_mis_rates.get(node, 1.0) * \
                self.node_to_count.get(node, node.ref_counter[gpu]) * \
                prefill_time(node.num_tokens, node.context_length)

    def migrate_node_cost(self, node: TreeNode, from_gpu, to_gpu):
        cost = self.per_node_prefill_cost[node]

        self.current_prefill_cost_per_gpu[from_gpu] -= cost
        self.current_prefill_cost_per_gpu[to_gpu] += cost
        
        total_decoding_length = self.per_node_total_decode_lengths[node]
        self.current_decode_lengths_per_gpu[from_gpu] -= total_decoding_length
        self.current_decode_lengths_per_gpu[to_gpu] += total_decoding_length



class GlobalSchedulerWithTime:
    def __init__(self, num_nodes=2, enable_eviction=False, enable_rebalancing=True, enable_miss_rate=True) -> None:
        self.num_gpus = num_nodes
        self.gpu_allocations = {}
        self.counter = 0
        self.enable_eviction = enable_eviction
        self.per_gpu_load = {i: 0 for i in range(num_nodes)}
        self.all_gpus = set(range(num_nodes))

        self.mem_cost = [0 for _ in range(num_nodes)]
        self.num_gpus = num_nodes
        self.metrics_dict = []
        self.lock = threading.Lock()
        self.enable_miss_rate = enable_miss_rate
        self.avg_topt_per_gpu = [deque(maxlen=200) for _ in range(num_nodes)] 
        for i in range(num_nodes):
            self.avg_topt_per_gpu[i].append(.15)

        self.histogram = SlidingWindowHistogram(
            window_duration=timedelta(minutes=3), 
            gpu_allocations=self.gpu_allocations, 
            num_gpus=self.num_gpus, 
            enable_miss_rate=self.enable_miss_rate,
            avg_topt_per_gpu=self.avg_topt_per_gpu
        )
        self.cache = LPRadixCache(histogram=self.histogram, num_gpus=self.num_gpus, lock=self.lock)
        self.max_tokens_gpu = [198516 for _ in range(num_nodes)]
        self.HIGH_LOAD_THRESHOLD = 1.5
        self.overload_detector = TTFTWindowedOverloadedDetector(window_duration=timedelta(minutes=3))
        self.enable_rebalancing = enable_rebalancing
        self.last_rebalancing_time = datetime.now()
        self.min_rebalancing_interval = timedelta(seconds=10)
        
    # Consider Split nodes
    def handle_split_nodes_gpu_allocations(self, split_nodes, gpu_allocations):
        for child_node, new_node in split_nodes.items():
            # FIXME: this should be a deepcopy
            gpu_allocations[new_node] = gpu_allocations[child_node].copy()

    def handle_split_node_histogram(self, split_nodes: Dict[TreeNode, TreeNode]):
        for child, parent_node in split_nodes.items():
            if self.is_large_node(parent_node) and not self.is_large_node(child): # new node is parent is now larger
                self.histogram.rename_node(child, parent_node)
                parent_node.decode_length.extend(child.decode_length)
                child.decode_length = []
                for gpu in self.gpu_allocations.get(child, {}):
                    self.overload_detector.rename_node(child, parent_node, gpu)
            
            if self.is_large_node(child):
                for gpu in self.gpu_allocations.get(child, {}):
                    self.histogram.update_prefill_cost_per_node(child, gpu)

    # Recursively update get/update parent gpu allocation
    def get_parent_gpu_allocation(self, node: TreeNode):
        if not node:
            return self.all_gpus
        if self.gpu_allocations.get(node):
            return self.gpu_allocations.get(node)
        return self.get_parent_gpu_allocation(node.parent)

    def update_gpu_cache_for_parent(self,node: TreeNode, gpu_id):
        if not node:
            return
        node.cached_gpus = node.cached_gpus.union(gpu_id)
        self.update_cached_gpus_of_parent(node.parent, gpu_id)

    def update_gpu_allocation_for_parent(self, node: TreeNode, gpu_id):
        if not node:
            return
        self.gpu_allocations[node] = self.gpu_allocations.get(node, set()).union(gpu_id)
        self.update_gpu_allocation_for_parent(node.parent, gpu_id)

    def is_small_node(self, node: TreeNode):
        return not self.is_large_node(node)

    def is_large_node(self, node: TreeNode):
        return node.num_tokens > node.context_so_far

    def get_important_node(self, node: TreeNode):
        if self.is_large_node(node):
            return node
        return self.get_important_node(node.parent)

    def evict_callback(self, node: TreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        # TODO: Maybe update the parent if child has no parent
        node.evicted_gpus.add(runtime_selected)
        node.cached_gpus.remove(runtime_selected)
        return len(node.value)

    def handle_eviction(self, runtime_selected):
        current_max_tokens = self.max_tokens_gpu[runtime_selected]
        assert self.cache.allocated_size(runtime_selected) >= 0
        if self.cache.allocated_size(runtime_selected) > current_max_tokens:
            num_tokens = self.cache.allocated_size(runtime_selected) - current_max_tokens
            self.cache.evict_with_runtime_id_without_removing(num_tokens, lambda node: self.evict_callback(node, runtime_selected), runtime_selected)
    
    # NOTE: simple heuristic used: assume GPU memory is always full
    #       -> evict size is the leaf node size
    def virtual_evict_for_routing(self, leaf_node: TreeNode, runtime_selected: int):
        num_to_evict = leaf_node.num_tokens
        evicted_tree_nodes = self.cache.virtual_lru_eviction(num_to_evict, runtime_selected)
        eviction_cost = 0
        victim: TreeNode
        for victim in evicted_tree_nodes:
            eviction_cost += self.histogram.get_eviction_prefill_cost(victim, runtime_selected, self.is_large_node(victim))
        return eviction_cost
    
    def calculate_min_load_cost(self, leaf_node, selected_gpus):
        # histogram_mem_cost = self.histogram.current_allocation_cost_per_gpu
        histogram_mem_cost = self.histogram.current_allocation_per_gpu()
        costs = []
        for gpu_id in selected_gpus:
            cost = histogram_mem_cost[gpu_id]
            if self.enable_eviction:
                cost += self.virtual_evict_for_routing(leaf_node, gpu_id)
            costs.append(cost)
        gpu_selected = int(np.argmin(costs))
        return gpu_selected


    def runtime_selector(
        self,
        text: str = None,
        request_id: str = None,
        input_ids=None,
        sampling_params=None,
        runtime_id_with_highest_hit_rate=None,
        *args, **kwargs,
    ):
        decoding_length = sampling_params.get("max_new_tokens", sampling_params.get("max_tokens", 45))
        # Tokenize the text
        start_time = time.time()
        with self.lock:
            split_nodes = {}
            leaf_node = self.cache.insert(tuple(input_ids), split_nodes=split_nodes)
            self.handle_split_nodes_gpu_allocations(split_nodes, self.gpu_allocations) # copies split node gpu allocation
            self.handle_split_node_histogram(split_nodes)

            important_node = self.get_important_node(leaf_node)
            if leaf_node.num_tokens < leaf_node.context_so_far: # check that gpu allocation exists for important node
                gpu_selected = self.get_parent_gpu_allocation(leaf_node)
                if len(gpu_selected) > 1:
                    runtime_idx = self.calculate_min_load_cost(leaf_node, gpu_selected)
                else:
                    runtime_idx = list(gpu_selected)[0]
            elif runtime_id_with_highest_hit_rate is not None:
                runtime_idx = runtime_id_with_highest_hit_rate
            else:
                runtime_idx = self.calculate_min_load_cost(leaf_node, selected_gpus=range(self.num_gpus))
            self.counter += 1
            self.update_gpu_allocation_for_parent(leaf_node, {runtime_idx}) # Updated gpu allocations up till parent
            self.cache.update_allocated_size(leaf_node, runtime_idx) # Update ref counters

            assert self.is_large_node(important_node)

            self.histogram.update(datetime.now(), important_node, leaf_node, runtime_idx, decoding_length=decoding_length)
            self.per_gpu_load[runtime_idx] += 1

            # NOTE: eviction handled by iterative feedback
            if self.enable_eviction:
                self.handle_eviction(runtime_idx)
            if self.enable_rebalancing:
                self.handle_important_node_stealing(runtime_idx)

        self.metrics_dict.append(
            {
                "text": text,
                "rid": request_id,
                "selected_runtime": runtime_idx,
                "overhead": time.time() - start_time,
            }
        )
        return runtime_idx
    
    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
    ):
        with self.lock:
            runtime_id = func_output.runtime_selected
            self.update_overload_detector(input_ids, runtime_id, func_output)
            important_node = self.get_important_node(self.cache.find_node(input_ids))
            # if func_output.output_len != 1:
            #     important_node.decode_length.append(func_output.output_len)
            self.cache.remove_completed_input_ids(input_ids, runtime_id)
            if func_output.tpot != 0 and func_output.output_len != 1:
                self.avg_topt_per_gpu[runtime_id].append(func_output.tpot)

            self.histogram.current_decode_lengths_per_gpu[runtime_id] -= func_output.max_new_tokens
            if important_node not in self.histogram.histogram:
                breakpoint()
            assert important_node in self.histogram.histogram
            self.histogram.per_node_total_decode_lengths[important_node] -= func_output.max_new_tokens

    def handle_important_node_stealing(self, scheduled_idx):
        if sum(self.per_gpu_load.values()) < 50 * self.num_gpus:
            return
        # allocation_cost_per_gpu = self.histogram.current_allocation_per_gpu_with_atleast_min_load(2)
        allocation_cost_per_gpu = self.histogram.current_allocation_per_gpu_with_atleast_min_load(2)
        allocations_with_indices = [(gpu_id, allocation_cost_per_gpu[gpu_id]) for gpu_id in range(len(allocation_cost_per_gpu))]
        # logger.info(allocations_with_indices)
        allocations_with_indices = list(sorted(allocations_with_indices, key=lambda x: -x[1]))
        self.handle_important_node_stealing_recursive(allocations_with_indices)

    def handle_important_node_stealing_recursive(self, allocation_cost_with_devices):
        if len(allocation_cost_with_devices) == 1:
            return
        larger_device, larger_allocation_cost = allocation_cost_with_devices[0]
        smaller_device, smaller_device_allocation_cost = allocation_cost_with_devices[-1] # Last element is the smallest

        if larger_allocation_cost < self.HIGH_LOAD_THRESHOLD * smaller_device_allocation_cost:
            return
        # if datetime.now() - self.last_rebalancing_time < self.min_rebalancing_interval:
        #     return
        # self.last_rebalancing_time = datetime.now()


        # if self.per_gpu_load[larger_device] < self.HIGH_LOAD_THRESHOLD * self.per_gpu_load[smaller_device]:
        #     return
        # Use a min heap to manage node costs
        tpot = np.median(self.avg_topt_per_gpu[larger_device])
        node_cost_for_gpu = []
        all_rebalancing_cost = []
        for node, cost in self.histogram.histogram.items():
            if larger_device in self.gpu_allocations.get(node) and self.is_large_node(node) and self.histogram.node_to_count[node] > 1:
                rebalancing_cost = self.histogram.get_node_cost(node, larger_device, tpot)
                heapq.heappush(node_cost_for_gpu, (rebalancing_cost, node))
                all_rebalancing_cost.append(rebalancing_cost)
        
        # pop from top of priority queue if it's smaller than 5 percentile of rest of the nodes
        # if all_rebalancing_cost:
        #     min_rebalancing_cost = np.percentile(all_rebalancing_cost, 5) # don't rebalance nodes that don't have very little cost
        #     breakpoint()
        #     while node_cost_for_gpu and node_cost_for_gpu[0][0] < min_rebalancing_cost:
        #         heapq.heappop(node_cost_for_gpu)

        if len(node_cost_for_gpu) == 1:
            # Handle load splitting a single node in two
            cost, node = node_cost_for_gpu[0] 
            cost /= 2 # load is now split into two
            # if not node.has_cached_gpu(smaller_device) and self.overload_detector.is_node_overloaded(node, larger_device): 
            if smaller_device not in self.gpu_allocations[node] and self.overload_detector.is_node_overloaded(node, larger_device): 
                # Copying the node to the smallest device will not change the larger allocation
                larger_allocation_cost -= cost
                smaller_device_allocation_cost += cost
                self.gpu_allocations[node].add(smaller_device)
                self.overload_detector.delete_after_allocation(node, larger_device)
        else:
            steal_n = 0
            while node_cost_for_gpu:
                node: TreeNode
                cost, node = heapq.heappop(node_cost_for_gpu)

                assert self.is_large_node(node)
                # if node.has_cached_gpu(smaller_device): # Avoid copying an existing device
                # if smaller_device in self.gpu_allocations[node]:
                #     continue

                if larger_allocation_cost - cost < smaller_device_allocation_cost + cost:
                    break
                larger_allocation_cost -= cost
                smaller_device_allocation_cost += cost
                self.gpu_allocations[node] = {smaller_device}

                self.histogram.migrate_node_cost(node, larger_device, smaller_device)
                self.update_children(node, smaller_device)
                steal_n += 1
                if larger_allocation_cost < self.HIGH_LOAD_THRESHOLD * smaller_device_allocation_cost:
                    break
            # Upstead the sorted allocation based on the new smallest allocation
            if steal_n != 0:
                logger.info(f"Steal {steal_n} nodes from {larger_device} to {smaller_device}")
        allocation_cost_with_devices[0] = (larger_device, larger_allocation_cost)
        allocation_cost_with_devices[-1] = (smaller_device, smaller_device_allocation_cost)
        self.handle_important_node_stealing_recursive(allocation_cost_with_devices[1:])

    def update_children(self, node: TreeNode, gpu_id):
        for child in node.children.values():
            self.gpu_allocations[child] = {gpu_id}
            self.update_children(child, gpu_id)
    
    def print(self):
        self._print_helper(self.cache.root_node, 0)

    def _print_helper(self, node: TreeNode, indent=0, depth=0):
        for key, child in node.children.items():
            allocated_gpus = self.gpu_allocations.get(child, set())
            print(f"{' ' * indent}{tokenizer.decode(child.value)[:20].strip()} Cached: {child.cached_gpus} Allocated: {allocated_gpus} Evicted: {child.evicted_gpus} {len(child.value)} Decode Lengths {list(child.decode_length)[:5]}")
            self._print_helper(child, indent=indent + 2, depth=depth + 1)

    def work_steal_low_loaded_prefixes(self):
        low_load_nodes = []
        allocation_cost_per_gpu = self.histogram.current_allocation_per_gpu_with_atleast_min_load_or_zero_ref(2)
        max_req = max(max(allocation_cost_per_gpu), 1)
        normalized_mem_cost = [x / max_req for x in allocation_cost_per_gpu]
        if self.counter < 20:
            return None
        for runtime_id, node in enumerate(normalized_mem_cost):
            if node < 0.1:
                low_load_nodes.append(runtime_id)
        if not low_load_nodes:
            return None
        y = len(low_load_nodes)
        x = self.num_gpus - y
        for runtime_id in low_load_nodes:
            if np.random.rand() < y/(x + y): # Steal the node

                return runtime_id
        return None

    def update_overload_detector(self, input_ids, runtime_idx, func_output: RequestFuncOutput):
        # Overload detector based on the current ttft
        leaf_node = self.cache.find_node(input_ids)
        important_node = self.get_important_node(leaf_node)
        self.overload_detector.add_data_point(datetime.now(), important_node, runtime_idx, func_output.ttft)
    