from collections import defaultdict
from datetime import datetime, timedelta
from greedy_lp import RequestFuncOutput
from mem_scheduler_evict_based_on_load_with_global import LPRadixCache, LPTreeNode
import time
import numpy as np
import threading
import heapq

from typing import List, Tuple
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

class SlidingWindowHistogram:
    def __init__(self, window_duration, num_buckets, num_gpus=2):
        self.window_duration = window_duration
        self.num_buckets = num_buckets
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
            for gpu in node.gpu_selections:
                allocation[gpu] += cost # potentionally divide by length of node.gpu_selections here
        return allocation

    def split_allocation_per_gpu(self):
        allocation = [0 for _ in range(self.num_gpus)]
        node: LPTreeNode

        # Keep a priority queue of gpu -> node sorted by cost
        for node, cost in self.histogram.items():
            for gpu in node.gpu_selections:
                allocation[gpu] += cost

        avg_allocation = sum(allocation) / self.num_gpus
        # while any allocation is above allocation replication threshold * 0.8:
            # split the highest code node into two. 
            # Sort gpus by their recomputation/prefix match to this node
            # if cost / 2 + current load > average load, then don't assign allocation to gpu. 
                # If this is true for all gpus break
            # update node.gpu_selections to include the node if it was true
            # Reinsert back into the priority queue with 1/2 the cost on each either gpu
        return allocation

# class HistogramBasedMemoryLoadScheduler:
#     def __init__(self, num_nodes=2) -> None:
#         self.mem_cost = [0 for _ in range(num_nodes)]
#         self.gpu_allocations = defaultdict(set)
#         self.num_gpus = num_nodes
#         self.metrics_dict = []
#         self.lock = threading.Lock()
#         self.histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), num_buckets=10, num_gpus=self.num_gpus)
#         self.cache = LPRadixCache(histogram=self.histogram, num_gpus=self.num_gpus)
#         self.max_tokens_gpu = [198516 for _ in range(num_nodes)]

#     def update_gpu_selections_of_parent(self,node: LPTreeNode, gpu_id):
#         if not node:
#             return
#         node.gpu_selections = node.gpu_selections.union(gpu_id)
#         self.update_gpu_selections_of_parent(node.parent, gpu_id)

#     def handle_split_nodes(self, split_nodes, gpu_allocations):
#         for k, v in split_nodes.items():
#             gpu_allocations[k] = gpu_allocations[v].copy()

#     def get_parent_gpu_selections(self, node: LPTreeNode):
#         if not node:
#             return set()
#         if node.gpu_selections:
#             return node.gpu_selections
#         return self.get_parent_gpu_selections(node.parent)

#     def is_large_node(self, node: LPTreeNode):
#         return node.num_tokens > node.context_length - node.num_tokens

#     def get_important_node(self, node: LPTreeNode):
#         if self.is_large_node(node):
#             return node
#         return self.get_important_node(node.parent)

#     def handle_eviction(self, runtime_selected):
#         current_max_tokens = self.max_tokens_gpu[runtime_selected]
#         if self.cache.allocated_size(runtime_selected) > current_max_tokens:
#             num_tokens = self.cache.allocated_size(runtime_selected) - current_max_tokens
#             self.cache.evict(num_tokens, lambda node: self.evict_callback(node, runtime_selected), runtime_selected)
#             print(f"GPU {runtime_selected} Evictable size: ", self.cache.allocated_size(runtime_selected), current_max_tokens)

#     def evict_callback(self, node: LPTreeNode, runtime_selected: int):
#         """Method to handle eviction logic."""
#         # TODO: Maybe update the parent if child has no parent
#         num_tokens = len(node.value)
#         # is_small_node = node.num_tokens < node.context_length - node.num_tokens
#         # self.mem_cost[runtime_selected] -= num_tokens
#         node.gpu_selections.remove(runtime_selected)
#         return len(node.value)

#     def runtime_selector(
#         self,
#         text: str = None,
#         request_id: str = None,
#         input_ids=None,
#         sampling_params=None
#     ):
#         # Tokenize the text
#         start_time = time.time()
#         with self.lock:
#             split_nodes = {}
#             leaf_node = self.cache.insert(tuple(input_ids), split_nodes=split_nodes)
#             self.handle_split_nodes(split_nodes, self.gpu_allocations)

#             for k, v in split_nodes.items():
#                 if not self.is_large_node(v): # the parent key is larger and more important. This should be in the histogram
#                     self.histogram.rename_node(v, k)
#             important_node = self.get_important_node(leaf_node)
#             self.histogram.update(datetime.now(), important_node, leaf_node)
#             if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens:
#                 gpu_selected = self.get_parent_gpu_selections(leaf_node)
#                 for gpu in gpu_selected:
#                     self.mem_cost[gpu] += leaf_node.context_length
#             else:
#                 # Current node is the important node
#                 recom_costs = []
#                 for gpu_id in range(self.num_gpus):
#                     recomputation_cost = leaf_node.context_length
#                     recom_costs.append(recomputation_cost)
#                 histogram_mem_cost = self.histogram.current_allocation_per_gpu()
#                 gpu_selected = int(np.argmin([recom_costs[gpu_id] + histogram_mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
#                 self.mem_cost[gpu_selected] += leaf_node.context_length
#                 gpu_selected = set([gpu_selected])
#             # assert self.histogram.current_allocation_per_gpu() == self.mem_cost
#             runtime_idx = list(gpu_selected)[0]

#             if len(gpu_selected) > 1:
#                 runtime_idx = int(np.random.choice(list(gpu_selected)))
#             self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
#             if len(leaf_node.gpu_selections) != 1:
#                 breakpoint()
#             self.cache.update_allocated_size(leaf_node, runtime_idx)
#             # self.handle_eviction(runtime_idx)

#         self.metrics_dict.append(
#             {
#                 "text": text,
#                 "rid": request_id,
#                 "selected_runtime": gpu_selected,
#                 "overhead": time.time() - start_time,
#             }
#         )

#         return int(runtime_idx)

    # def print(self):
    #     _print_helper(self.cache.root_node, 0)

#     def finish_request(
#         self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
#     ):
#         with self.lock:
            # pass
def _print_helper(node, indent=0, depth=0):
    for key, child in node.children.items():
        print(" " * indent, tokenizer.decode(child.value)[:20], child.gpu_selections, len(child.value))
        _print_helper(child, indent=indent + 2, depth=depth + 1)


class HistogramBasedRecomp:
    def __init__(self, num_nodes=2, enable_eviction=True) -> None:
        self.num_gpus = num_nodes
        self.gpu_allocations = defaultdict(set)
        self.counter = 0
        self.enable_eviction = enable_eviction
        self.per_gpu_load = {i: 0 for i in range(num_nodes)}

        self.mem_cost = [0 for _ in range(num_nodes)]
        self.num_gpus = num_nodes
        self.metrics_dict = []
        self.lock = threading.Lock()
        self.histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), num_buckets=10, num_gpus=self.num_gpus)
        self.cache = LPRadixCache(histogram=self.histogram, num_gpus=self.num_gpus)
        self.max_tokens_gpu = [198516 for _ in range(num_nodes)]

    def update_gpu_selections_of_parent(self,node: LPTreeNode, gpu_id):
        if not node:
            return
        node.gpu_selections = node.gpu_selections.union(gpu_id)
        self.update_gpu_selections_of_parent(node.parent, gpu_id)

    def handle_split_nodes(self, split_nodes, gpu_allocations):
        for k, v in split_nodes.items():
            gpu_allocations[k] = gpu_allocations[v].copy()

    def get_parent_gpu_selections(self, node: LPTreeNode):
        if not node:
            return set()
        if node.gpu_selections:
            return node.gpu_selections
        return self.get_parent_gpu_selections(node.parent)

    def print(self):
        _print_helper(self.cache.root_node, 0)

    def is_large_node(self, node: LPTreeNode):
        return node.num_tokens > node.context_length - node.num_tokens

    def get_important_node(self, node: LPTreeNode):
        if self.is_large_node(node):
            return node
        return self.get_important_node(node.parent)

    def get_recomp_cost(self, node: LPTreeNode, gpu_id, histogram: SlidingWindowHistogram):
        if not node or gpu_id in node.gpu_selections and not node.is_evicted:
            return 0
        else:
            return histogram.histogram.get(node, 0) + self.get_recomp_cost(node.parent, gpu_id, histogram)
    
    def get_recomp_cost_basic(self, node: LPTreeNode, gpu_id, histogram: SlidingWindowHistogram):
        if not node or (gpu_id in node.gpu_selections and not node.is_evicted):
            return 0
        else:
            return node.num_tokens + self.get_recomp_cost_basic(node.parent, gpu_id, histogram)
    

    def evict_callback(self, node: LPTreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        # TODO: Maybe update the parent if child has no parent
        num_tokens = len(node.value)
        # is_small_node = node.num_tokens < node.context_length - node.num_tokens
        # self.mem_cost[runtime_selected] -= num_tokens
        # if runtime_selected not in node.gpu_selections: # Caused by migration
        #     return 0
        # node.gpu_selections.remove(runtime_selected)
        self.cache.allocated_size_[runtime_selected] -= len(node.value)
        node.is_evicted = True
        # If parent doesn't have any runtime selected anymore. Remove it
        # curr_node = node.parent
        # while curr_node is not None:
        #     # curr_node.parent = Union of all the children's gpu selection sets
        #     gpu_selections = set()
        #     for child in curr_node.children.values():
        #         gpu_selections = curr_node.gpu_selections.union(child.gpu_selections)
        #     curr_node.gpu_selections = gpu_selections
        #     curr_node = curr_node.parent
        return len(node.value)

    def handle_eviction(self, runtime_selected):
        current_max_tokens = self.max_tokens_gpu[runtime_selected]
        assert self.cache.allocated_size(runtime_selected) >= 0
        if self.cache.allocated_size(runtime_selected) > current_max_tokens:
            # breakpoint()
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
            self.handle_split_nodes(split_nodes, self.gpu_allocations)

            for child, parent_node in split_nodes.items():
                if self.is_large_node(parent_node) and not self.is_large_node(child): # new node is parent is now larger
                    self.histogram.rename_node(child, parent_node)
            important_node = self.get_important_node(leaf_node)
            # smth is bigger than the path so far, treat as important. since greedy can be unstable, schedule based on this fact
            if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens and important_node.gpu_selections: # This is saying that in case of eviction resolve
                gpu_selected = self.get_parent_gpu_selections(leaf_node)
            else:
                recom_costs = []
                for gpu_id in range(self.num_gpus):
                    recomputation_cost = self.get_recomp_cost(leaf_node.parent, gpu_id, self.histogram)
                    recom_costs.append(recomputation_cost)
                histogram_mem_cost = self.histogram.current_allocation_per_gpu()
                gpu_selected = int(np.argmin([recom_costs[gpu_id] + histogram_mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
                gpu_selected = set([gpu_selected])
            # self.update_gpu_selections_of_parent(leaf_node, gpu_selected)
            self.histogram.update(datetime.now(), important_node, leaf_node)
            runtime_idx = list(gpu_selected)[0]

            if len(gpu_selected) > 1:
                runtime_idx = int(np.random.choice(list(gpu_selected)))
            self.per_gpu_load[runtime_idx] += 1

            self.cache.update_allocated_size(leaf_node, runtime_idx)
            self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
    
            current_allocated_size = 0
            for node in self.cache._collect_nodes():
                node: LPTreeNode
                if runtime_idx in node.gpu_selections and not node.is_evicted:
                    current_allocated_size += len(node.value)
            # if current_allocated_size != self.cache.allocated_size(runtime_idx):
            #     breakpoint() TODO FIXME
            # assert current_allocated_size == self.cache.allocated_size(runtime_idx)
            if self.enable_eviction:
                self.handle_eviction(runtime_idx)

            self.counter += 1    
            if self.counter % 500:
                print(self.per_gpu_load)
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

        allocation = self.histogram.current_allocation_per_gpu()
        # TODO handling the case of two gpus
        # Determine the larger and smaller allocations
        larger_index, smaller_index = (0, 1) if allocation[0] > allocation[1] else (1, 0)
        # Check if the larger allocation is significantly bigger
        if self.per_gpu_load[larger_index] > 1.5 * self.per_gpu_load[smaller_index]:
            print("Stealing from ", larger_index, " to ", smaller_index)
            # Use a min heap to manage node costs
            node_cost_for_gpu = []
            for node, cost in self.histogram.histogram.items():
                # If after adjusting the nodes, the allocation difference is valid, allow adjustment
                if larger_index in node.gpu_selections:
                    heapq.heappush(node_cost_for_gpu, (cost, node))
            while node_cost_for_gpu:
                node: LPTreeNode
                cost, node = heapq.heappop(node_cost_for_gpu)
                if len(node.gpu_selections) != 1:
                    breakpoint()
                assert len(node.gpu_selections) == 1
                if not self.is_large_node(node):
                    breakpoint()
                    assert False
                if allocation[larger_index] < allocation[smaller_index] + cost: # should I add an edge case to check if larger_node is no longer much smaller than this node??
                    break
                allocation[larger_index] -= cost
                allocation[smaller_index] += cost
                node.gpu_selections = {smaller_index} # Move gpu assignment to other gpu
                # node.last_access_time = time.time()
                self.update_children(node, smaller_index)
                self.update_gpu_selections_of_parent(node, {smaller_index})
                self.cache.migrate_allocated_size(node, smaller_index, larger_index)

    def update_children(self, node: LPTreeNode, gpu_id):
        for child in node.children.values():
            child.gpu_selections = {gpu_id}
            self.update_children(child, gpu_id)

    def update_gpu_selections_of_parent(self,node: LPTreeNode, gpu_id):
        node.gpu_selections = node.gpu_selections.union(gpu_id)
        node: LPTreeNode = node.parent
        while node:
            gpu_selections = set()
            for child in node.children.values():
                gpu_selections = gpu_selections.union(child.gpu_selections)
            node.gpu_selections = gpu_selections
            node = node.parent
