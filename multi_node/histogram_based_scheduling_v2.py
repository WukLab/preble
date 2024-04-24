from collections import defaultdict
from datetime import datetime, timedelta
from greedy_lp import RequestFuncOutput
from global_lru_cache import LPRadixCache, LPTreeNode
import time
import numpy as np
import threading
import heapq

from typing import List, Tuple
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

class SlidingWindowHistogram:
    def __init__(self, window_duration: timedelta, gpu_allocations, num_gpus=2):
        self.window_duration = window_duration
        self.gpu_allocations = gpu_allocations
        self.histogram = defaultdict(int)
        self.timestamps: List[Tuple[datetime, LPTreeNode, LPTreeNode]] = []
        self.num_gpus = num_gpus

    def update(self, timestamp, node: LPTreeNode, leaf_node: LPTreeNode):
        self.timestamps.append((timestamp, node, leaf_node))
        self.histogram[node] += 1 * leaf_node.context_length
        self._remove_old_entries(timestamp)

    def _remove_old_entries(self, current_timestamp):
        window_start = current_timestamp - self.window_duration
        while self.timestamps and self.timestamps[0][0] < window_start:
            timestamp, node, leaf_node = self.timestamps.pop(0)
            self.histogram[node] -= 1 * leaf_node.context_length
            if self.histogram[node] <= 0:
                del self.histogram[node]
                self.gpu_allocations[node] = set() # Reset the gpu allocation outside the time window

    def rename_node(self, old_node, new_node):
        if old_node in self.histogram:
            self.histogram[new_node] = self.histogram.pop(old_node)
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
        node: LPTreeNode
        for node, cost in self.histogram.items():
            for gpu in self.gpu_allocations.get(node, {}):
                allocation[gpu] += cost / len(self.gpu_allocations.get(node))# potentionally divide by length of node.cached_gpus here
        return allocation



class HistogramBasedRecompV2:
    def __init__(self, num_nodes=2, enable_eviction=True) -> None:
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
        self.histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), gpu_allocations=self.gpu_allocations, num_gpus=self.num_gpus)
        self.cache = LPRadixCache(histogram=self.histogram, num_gpus=self.num_gpus)
        self.max_tokens_gpu = [198516 for _ in range(num_nodes)]
        self.HIGH_LOAD_THRESHOLD = 1.5

    # Consider Split nodes
    def handle_split_nodes_gpu_allocations(self, split_nodes, gpu_allocations):
        for child_node, new_node in split_nodes.items():
            gpu_allocations[new_node] = gpu_allocations[child_node].copy()

    def handle_split_node_histogram(self, split_nodes):
        for child, parent_node in split_nodes.items():
            if self.is_large_node(parent_node) and not self.is_large_node(child): # new node is parent is now larger
                self.histogram.rename_node(child, parent_node)

    # Recursively update get/update parent gpu allocation
    def get_parent_gpu_allocation(self, node: LPTreeNode):
        if not node:
            return self.all_gpus
        if self.gpu_allocations.get(node):
            return self.gpu_allocations.get(node)
        return self.get_parent_gpu_allocation(node.parent)

    def update_gpu_cache_for_parent(self,node: LPTreeNode, gpu_id):
        if not node:
            return
        node.cached_gpus = node.cached_gpus.union(gpu_id)
        self.update_cached_gpus_of_parent(node.parent, gpu_id)

    def update_gpu_allocation_for_parent(self, node: LPTreeNode, gpu_id):
        if not node:
            return
        self.gpu_allocations[node] = self.gpu_allocations.get(node, set()).union(gpu_id)
        self.update_gpu_allocation_for_parent(node.parent, gpu_id)

    def is_small_node(self, node: LPTreeNode):
        return not self.is_large_node(node)

    def is_large_node(self, node: LPTreeNode):
        return node.num_tokens > node.context_length - node.num_tokens

    def get_important_node(self, node: LPTreeNode):
        if self.is_large_node(node):
            return node
        return self.get_important_node(node.parent)

    def get_recomp_cost(self, node: LPTreeNode, gpu_id, histogram: SlidingWindowHistogram):
        if not node:
            return 0
        if node.has_cached_gpu(gpu_id):
            return 0
        else:
            return histogram.histogram.get(node, 1) + self.get_recomp_cost(node.parent, gpu_id, histogram)

    def get_recomp_cost_basic(self, node: LPTreeNode, gpu_id):
        if not node:
            return 0
        if node.has_cached_gpu(gpu_id):
            return 0
        else:
            return node.num_tokens * node.ref_counter + self.get_recomp_cost_basic(node.parent, gpu_id)

    def evict_callback(self, node: LPTreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        # TODO: Maybe update the parent if child has no parent
        self.cache.allocated_size_[runtime_selected] -= len(node.value)
        node.evicted_gpus.add(runtime_selected)
        node.cached_gpus.remove(runtime_selected)
        return len(node.value)

    def handle_eviction(self, runtime_selected):
        current_max_tokens = self.max_tokens_gpu[runtime_selected]
        assert self.cache.allocated_size(runtime_selected) >= 0
        if self.cache.allocated_size(runtime_selected) > current_max_tokens:
            num_tokens = self.cache.allocated_size(runtime_selected) - current_max_tokens
            self.cache.evict_with_runtime_id_without_removing(num_tokens, lambda node: self.evict_callback(node, runtime_selected), runtime_selected)
            print(f"GPU {runtime_selected} Evictable size: ", self.cache.allocated_size(runtime_selected), current_max_tokens)

    def runtime_selector(
        self,
        text: str = None,
        request_id: str = None,
        input_ids=None,
        sampling_params=None
    ):
        # Tokenize the text
        start_time = time.time()
        with self.lock:
            split_nodes = {}
            leaf_node = self.cache.insert(tuple(input_ids), split_nodes=split_nodes)
            self.handle_split_nodes_gpu_allocations(split_nodes, self.gpu_allocations) # copies split node gpu allocation
            self.handle_split_node_histogram(split_nodes)

            important_node = self.get_important_node(leaf_node)
            if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens: # check that gpu allocation exists for important node
                gpu_selected = self.get_parent_gpu_allocation(leaf_node)
            else:
                recom_costs = []
                for gpu_id in range(self.num_gpus):
                    recomputation_cost = self.get_recomp_cost_basic(leaf_node.parent, gpu_id)
                    recom_costs.append(recomputation_cost)
                histogram_mem_cost = self.histogram.current_allocation_per_gpu()
                gpu_selected = int(np.argmin([recom_costs[gpu_id] + histogram_mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
                gpu_selected = set([gpu_selected])

            self.histogram.update(datetime.now(), important_node, leaf_node)
            self.update_gpu_allocation_for_parent(leaf_node, gpu_selected)
            runtime_idx = list(gpu_selected)[0]
            if len(gpu_selected) > 1:
                runtime_idx = int(np.random.choice(list(gpu_selected)))

            self.per_gpu_load[runtime_idx] += 1
            self.cache.update_allocated_size(leaf_node, runtime_idx)
    
            current_allocated_size = 0
            for node in self.cache._collect_nodes():
                node: LPTreeNode
                if node.has_cached_gpu(runtime_idx):
                    current_allocated_size += len(node.value)
            # if current_allocated_size != self.cache.allocated_size(runtime_idx):
            #     breakpoint() TODO FIXME
            # assert current_allocated_size == self.cache.allocated_size(runtime_idx)
            # if self.enable_eviction:
            #     self.handle_eviction(runtime_idx)

            self.counter += 1    
            if self.counter % 500:
                print(self.per_gpu_load)
            self.handle_work_stealing(runtime_idx)

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
            self.cache.remove_completed_input_ids(input_ids)

    def handle_important_node_stealing(self, scheduled_idx):
        if sum(self.per_gpu_load.values()) < 50:
            return
        allocation_cost_per_gpu = self.histogram.current_allocation_per_gpu()
        allocations_with_indices = [(gpu_id, allocation_cost_per_gpu[gpu_id]) for gpu_id in range(len(allocation_cost_per_gpu))]
        allocations_with_indices = list(sorted(allocations_with_indices, key=lambda x: -x[1]))
        self.handle_important_node_stealing_recursive(allocations_with_indices)

    def handle_important_node_stealing_recursive(self, allocation_cost_with_devices):
        if len(allocation_cost_with_devices) == 1:
            return
        larger_device, larger_allocation_cost = allocation_cost_with_devices[0]
        smaller_device, smaller_device_allocation_cost = allocation_cost_with_devices[-1] # Last element is the smallest

        if larger_allocation_cost < smaller_device_allocation_cost:
            return

        if self.per_gpu_load[larger_device] < self.HIGH_LOAD_THRESHOLD * self.per_gpu_load[smaller_device]:
            return
        
        # Use a min heap to manage node costs
        node_cost_for_gpu = []
        for node, cost in self.histogram.histogram.items():
            # If after adjusting the nodes, the allocation difference is valid, allow adjustment
            if larger_device in self.gpu_allocations.get(node):
                heapq.heappush(node_cost_for_gpu, (cost, node))
        
        if len(node_cost_for_gpu) == 1:
            # Handle load splitting a single node in two
            cost, node = node_cost_for_gpu[0] 
            cost /= 2 # load is now split into two

            if not node.has_cached_gpu(smaller_device) and larger_allocation_cost - cost > smaller_device_allocation_cost + cost: 
                # Copying the node to the smallest device will not change the larger allocation
                larger_allocation_cost -= cost
                smaller_device_allocation_cost += cost
                self.gpu_allocations[node].add(smaller_device)
        else:
            while node_cost_for_gpu:
                node: LPTreeNode
                cost, node = heapq.heappop(node_cost_for_gpu)

                assert self.is_large_node(node)
                if node.has_cached_gpu(smaller_device): # Avoid copying an existing device
                    continue

                if larger_allocation_cost - cost < smaller_device_allocation_cost + cost:
                    break
                larger_allocation_cost -= cost
                smaller_device_allocation_cost += cost
                self.gpu_allocations[node] = {smaller_device}
                self.update_children(node, smaller_device)
            # Upstead the sorted allocation based on the new smallest allocation
        allocation_cost_with_devices[0] = (larger_device, larger_allocation_cost)
        allocation_cost_with_devices[-1] = (smaller_device, smaller_device_allocation_cost)
        self.handle_important_node_stealing_recursive(allocation_cost_with_devices[1:])

    def update_children(self, node: LPTreeNode, gpu_id):
        for child in node.children.values():
            self.gpu_allocations[child] = {gpu_id}
            self.update_children(child, gpu_id)
    
    def print(self):
        self._print_helper(self.cache.root_node, 0)

    def _print_helper(self, node: LPTreeNode, indent=0, depth=0):
        for key, child in node.children.items():
            allocated_gpus = self.gpu_allocations.get(child, set())
            print(f"{' ' * indent}{tokenizer.decode(child.value)[:20].strip()} Cached: {child.cached_gpus} Allocated: {allocated_gpus} {len(child.value)}")
            self._print_helper(child, indent=indent + 2, depth=depth + 1)

    def work_steal_low_loaded_prefixes(self):
        low_load_nodes = []
        max_req = max(max(self.mem_cost), 1)
        normalized_mem_cost = [x / max_req for x in self.mem_cost]
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
