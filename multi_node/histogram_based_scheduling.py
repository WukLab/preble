from collections import defaultdict
from datetime import datetime, timedelta
from greedy_lp import LPTreeNode, LPRadixCache, RequestFuncOutput
import time
import numpy as np
import threading
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
            self.histogram[new_node] = self.histogram.pop(new_node)
            rename_mapping = {old_node: new_node}
            self.timestamps = [(timestamp, rename_mapping.get(node, node)) for timestamp, node in self.timestamps]

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

class HistogramBasedMemoryLoadScheduler:
    def __init__(self, num_nodes=2) -> None:
        self.mem_cost = [0 for _ in range(num_nodes)]
        self.gpu_allocations = defaultdict(set)
        self.num_gpus = num_nodes
        self.metrics_dict = []
        self.lock = threading.Lock()
        self.cache = LPRadixCache()
        # self.hisotgram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), num_buckets=10, num_gpus=self.num_gpus)
        self.hisotgram = SlidingWindowHistogram(window_duration=400, num_buckets=10, num_gpus=self.num_gpus)

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

    def is_large_node(self, node: LPTreeNode):
        return node.num_tokens > node.context_length - node.num_tokens

    def get_important_node(self, node: LPTreeNode):
        if self.is_large_node(node):
            return node
        return self.get_important_node(node.parent)

    def runtime_selector(
        self,
        text: str = None,
        request_id: str = None,
        input_ids=None,
        sampling_params=None,
        current_time_stamp=None, # for simulation clock
    ):
        # Tokenize the text
        start_time = time.time()
        with self.lock:
            split_nodes = {}
            leaf_node = self.cache.insert(tuple(input_ids), split_nodes=split_nodes)
            self.handle_split_nodes(split_nodes, self.gpu_allocations)

            for k, v in split_nodes.items():
                if not self.is_large_node(v): # the parent key is larger and more important. This should be in the histogram
                    self.hisotgram.rename_node(v, k)
            important_node = self.get_important_node(leaf_node)
            if current_time_stamp is None:
                current_time_stamp = datetime.now()
            self.hisotgram.update(current_time_stamp, important_node, leaf_node)
            if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens:
                gpu_selected = self.get_parent_gpu_selections(leaf_node)
                # for gpu in gpu_selected:
                #     self.mem_cost[gpu] += leaf_node.context_length
            else:
                # Current node is the important node
                recom_costs = []
                for gpu_id in range(self.num_gpus):
                    recomputation_cost = leaf_node.context_length
                    recom_costs.append(recomputation_cost)
                histogram_mem_cost = self.hisotgram.current_allocation_per_gpu()
                gpu_selected = int(np.argmin([recom_costs[gpu_id] + histogram_mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
                self.mem_cost[gpu_selected] += leaf_node.context_length
                gpu_selected = set([gpu_selected])
            self.update_gpu_selections_of_parent(leaf_node, gpu_selected)
            # assert self.hisotgram.current_allocation_per_gpu() == self.mem_cost
        self.metrics_dict.append(
            {
                "text": text,
                "rid": request_id,
                "selected_runtime": gpu_selected,
                "overhead": time.time() - start_time,
            }
        )
        if len(gpu_selected) > 1:
            random_gpu = np.random.choice(list(gpu_selected))
            self.mem_cost[random_gpu] += leaf_node.context_length
            return int(random_gpu)
        runtime_idx = list(gpu_selected)[0]
        return int(runtime_idx)

    def print(self):
        _print_helper(self.cache.root_node, 0)

    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
    ):
        with self.lock:
            pass
def _print_helper(node, indent=0, depth=0):
    for key, child in node.children.items():
        print(" " * indent, tokenizer.decode(child.value)[:20], child.gpu_selections, len(child.value))
        _print_helper(child, indent=indent + 2, depth=depth + 1)


class HistogramBasedRecomp(HistogramBasedMemoryLoadScheduler):
    def __init__(self, num_nodes=2) -> None:
        super().__init__(num_nodes)
        self.gpu_allocations = defaultdict(set)

    def get_recomp_cost(self, node: LPTreeNode, gpu_id, histogram: SlidingWindowHistogram):
        if not node or gpu_id in node.gpu_selections:
            return 0
        else:
            return node.num_tokens * histogram.histogram.get(node, 0) + self.get_recomp_cost(node.parent, gpu_id, histogram)

    def runtime_selector(
        self,
        text: str = None,
        request_id: str = None,
        input_ids=None,
        sampling_params=None,
        current_time_stamp=None,
    ):
        # Tokenize the text
        start_time = time.time()
        with self.lock:
            split_nodes = {}
            leaf_node = self.cache.insert(tuple(input_ids), split_nodes=split_nodes)
            self.handle_split_nodes(split_nodes, self.gpu_allocations)

            for k, v in split_nodes.items():
                if not self.is_large_node(v): # the parent key is larger and more important. This should be in the histogram
                    self.hisotgram.rename_node(v, k)

            important_node = self.get_important_node(leaf_node)
            if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens:
                gpu_selected = self.get_parent_gpu_selections(leaf_node)
                # for gpu in gpu_selected:
                #     self.mem_cost[gpu] += leaf_node.context_length
            else:
                # Current node is the important node
                recom_costs = []
                for gpu_id in range(self.num_gpus):
                    recomputation_cost = self.get_recomp_cost(leaf_node.parent, gpu_id, self.hisotgram) + leaf_node.context_length
                    recom_costs.append(recomputation_cost)
                histogram_mem_cost = self.hisotgram.current_allocation_per_gpu()
                gpu_selected = int(np.argmin([recom_costs[gpu_id] + histogram_mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
                self.mem_cost[gpu_selected] += leaf_node.context_length
                gpu_selected = set([gpu_selected])
            self.update_gpu_selections_of_parent(leaf_node, gpu_selected)
            if current_time_stamp is None:
                current_time_stamp = datetime.now()
            self.hisotgram.update(current_time_stamp, important_node, leaf_node)
        self.metrics_dict.append(
            {
                "text": text,
                "rid": request_id,
                "selected_runtime": gpu_selected,
                "overhead": time.time() - start_time,
            }
        )
        if len(gpu_selected) > 1:
            random_gpu = np.random.choice(list(gpu_selected))
            return int(random_gpu)
        runtime_idx = list(gpu_selected)[0]
        return int(runtime_idx)
