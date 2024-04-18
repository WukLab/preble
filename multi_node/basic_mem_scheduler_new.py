from collections import defaultdict
import time
import numpy as np
from greedy_lp import LPTreeNode, LPRadixCache, RequestFuncOutput
import threading
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def _print_helper(node, indent, depth=0):
    for key, child in node.children.items():
        print(" " * indent, tokenizer.decode(child.value)[:20], child.gpu_selections, len(child.value))
        _print_helper(child, indent=indent + 2, depth=depth + 1)

class BasicMemScheduler:
    def __init__(self, num_nodes=2) -> None:
        self.mem_cost = [0 for _ in range(num_nodes)]
        self.gpu_allocations = defaultdict(set)
        self.num_gpus = num_nodes
        self.lock = threading.Lock()
        self.cache = LPRadixCache()
        self.metrics_dict = []

    def get_recomp_cost(self, node: LPTreeNode, gpu_id):
        if not node or gpu_id in self.gpu_allocations[node]:
            return 0
        else:
            return node.num_tokens + self.get_recomp_cost(node.parent, gpu_id)

    def update_gpu_selections_of_parent(self,node: LPTreeNode, gpu_id):
        if not node:
            return
        node.gpu_selections.add(gpu_id)
        self.update_gpu_selections_of_parent(node.parent, gpu_id)

    def handle_split_nodes(self, split_nodes, gpu_allocations):
        for k, v in split_nodes.items():
            gpu_allocations[k] = gpu_allocations[v].copy()

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
            recom_costs = []
            for gpu_id in range(self.num_gpus):
                recomputation_cost = self.get_recomp_cost(leaf_node, gpu_id)
                recom_costs.append(recomputation_cost)
            gpu_selected = int(np.argmin([recom_costs[gpu_id] + self.mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
            self.mem_cost[gpu_selected] += recom_costs[gpu_selected]
            self.update_gpu_selections_of_parent(leaf_node, gpu_selected)

        self.metrics_dict.append(
            {
                "text": text,
                "rid": request_id,
                "selected_runtime": gpu_selected,
                "overhead": time.time() - start_time,
            }
        )
        print(f"Mem Cost on each gpu:", self.mem_cost)
        return gpu_selected

    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
    ):
        with self.lock:
            pass

    def print(self):
        _print_helper(self.cache.root_node, 0)


# class BasicMemSchedulerV2:
#     def __init__(self, num_nodes=2) -> None:
#         self.mem_cost = [0 for _ in range(num_nodes)]
#         self.gpu_allocations = defaultdict(set)
#         self.num_gpus = num_nodes
#         self.lock = threading.Lock()
#         self.cache = LPRadixCache()
#         self.metrics_dict = []

#     def get_recomp_cost(self, node: LPTreeNode, gpu_id):
#         if not node or gpu_id in self.gpu_allocations[node]:
#             return 0
#         else:
#             return node.num_tokens + self.get_recomp_cost(node.parent, gpu_id)

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
#             if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens:
#                 gpu_selected = self.get_parent_gpu_selections(leaf_node)
#                 for gpu in gpu_selected:
#                     self.mem_cost[gpu] += self.get_recomp_cost(leaf_node, gpu)
#             else:
#                 recom_costs = []
#                 for gpu_id in range(self.num_gpus):
#                     recomputation_cost = self.get_recomp_cost(leaf_node, gpu_id)
#                     recom_costs.append(recomputation_cost)
#                 gpu_selected = int(np.argmin([recom_costs[gpu_id] + self.mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
#                 self.mem_cost[gpu_selected] += recom_costs[gpu_selected]
#                 gpu_selected = set([gpu_selected])
#             self.update_gpu_selections_of_parent(leaf_node, gpu_selected)

#         self.metrics_dict.append(
#             {
#                 "text": text,
#                 "rid": request_id,
#                 "selected_runtime": gpu_selected,
#                 "overhead": time.time() - start_time,
#             }
#         )
#         if len(gpu_selected) > 1:
#             random_gpu = np.random.choice(list(gpu_selected))
#             return int(random_gpu)
#         runtime_idx = list(gpu_selected)[0]

#         return int(runtime_idx)

#     def finish_request(
#         self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
#     ):
#         with self.lock:
#             pass

#     def print(self):
#         _print_helper(self.cache.root_node, 0)


class BasicMemSchedulerV3:
    def __init__(self, num_nodes=2) -> None:
        self.mem_cost = [0 for _ in range(num_nodes)]
        self.gpu_allocations = defaultdict(set)
        self.num_gpus = num_nodes
        self.lock = threading.Lock()
        self.cache = LPRadixCache()
        self.metrics_dict = []
        self.runtime_caches = [LPRadixCache() for _ in range(num_nodes)]
        self.max_tokens_gpu = [198516, 198516]
        self.counter = 0

    def get_recomp_cost(self, node: LPTreeNode, gpu_id):
        if not node or gpu_id in node.gpu_selections:
            return 0
        else:
            return node.num_tokens + self.get_recomp_cost(node.parent, gpu_id)

    def update_gpu_selections_of_parent(self,node: LPTreeNode, gpu_id):
        if not node:
            return
        node.gpu_selections = node.gpu_selections.union(gpu_id)
        self.update_gpu_selections_of_parent(node.parent, gpu_id)

    def get_parent_gpu_selections(self, node: LPTreeNode):
        if not node:
            return set()
        if node.gpu_selections:
            return node.gpu_selections
        return self.get_parent_gpu_selections(node.parent)

    def insert_then_evict_from_runtime_cache(self, input_ids, runtime_selected):
        runtime_cache = self.runtime_caches[runtime_selected]
        node = runtime_cache.insert(tuple(input_ids))
        current_max_tokens = self.max_tokens_gpu[runtime_selected]
        if runtime_cache.evictable_size() > current_max_tokens:
            num_tokens = runtime_cache.evictable_size() - current_max_tokens
            runtime_cache.evict(num_tokens, lambda node: self.evict_callback(node, runtime_selected))
            print(f"GPU {runtime_selected} Evictable size: ", runtime_cache.evictable_size(), current_max_tokens)


    def evict_callback(self, node: LPTreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        updated_node = self.gpu_allocations.get(node)
        if updated_node:
            updated_node.remove(runtime_selected)
            if len(updated_node) == 0:
                del self.gpu_allocations[node]
        num_tokens = len(node.value)
        self.mem_cost[runtime_selected] -= num_tokens
        return len(node.value)

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
            recom_costs = []
            for gpu_id in range(self.num_gpus):
                recomputation_cost = self.get_recomp_cost(leaf_node.parent, gpu_id)
                recom_costs.append(recomputation_cost)
            is_small_node = leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens
            if is_small_node:
                gpu_selected = self.get_parent_gpu_selections(leaf_node.parent)
                assert len(gpu_selected) == 1
                runtime_idx = list(gpu_selected)[0]
                self.mem_cost[runtime_idx] += recom_costs[runtime_idx] + leaf_node.num_tokens
                self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
            else:
                cost_f = lambda gpu_id: recom_costs[gpu_id] + self.mem_cost[gpu_id] + leaf_node.num_tokens
                gpu_selected = min(range(self.num_gpus), key=cost_f)
                gpu_selected = set([gpu_selected])
                runtime_idx = list(gpu_selected)[0]
                self.mem_cost[runtime_idx] += recom_costs[runtime_idx] + leaf_node.num_tokens
                self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
            # self.mem_cost[runtime_idx] += recom_costs[runtime_idx]
            # Maybe memory cost should only be updated for the one that gets selected not scheduled
            self.counter += 1
            self.insert_then_evict_from_runtime_cache(input_ids, runtime_idx)
            self.metrics_dict.append(
                {
                    "text": text[:100],
                    "rid": request_id,
                    "selected_runtime": runtime_idx,
                    "overhead": time.time() - start_time,
                    "mem_costs": self.mem_cost,
                    "parent_memory_cost": self.get_recomp_cost(leaf_node.parent, runtime_idx),
                    "current_leaf_node_cost": leaf_node.num_tokens,
                }
            )

            return int(runtime_idx)

    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
    ):
        with self.lock:
            pass
            # self.cache.remove_completed_input_ids(input_ids)
            # self.runtime_caches[func_output.runtime_selected].remove_completed_input_ids(input_ids)

    def print(self):
        _print_helper(self.cache.root_node, 0)

def _print_helper(node, indent, depth=0):
    for key, child in node.children.items():
        print(" " * indent, tokenizer.decode(child.value)[:20], child.gpu_selections, len(child.value))
        _print_helper(child, indent=indent + 2, depth=depth + 1)


class BasicMemSchedulerV4:
    def __init__(self, num_nodes=2) -> None:
        self.mem_cost = [0 for _ in range(num_nodes)]
        self.gpu_allocations = defaultdict(set)
        self.num_gpus = num_nodes
        self.lock = threading.Lock()
        self.cache = LPRadixCache()
        self.metrics_dict = []
        self.runtime_caches = [LPRadixCache() for _ in range(num_nodes)]
        self.max_tokens_gpu = [198516, 198516]
        self.counter = 0

    def get_recomp_cost(self, node: LPTreeNode, gpu_id):
        if not node or gpu_id in node.gpu_selections:
            return 0
        else:
            return node.num_tokens + self.get_recomp_cost(node.parent, gpu_id)

    def update_gpu_selections_of_parent(self,node: LPTreeNode, gpu_id):
        if not node:
            return
        node.gpu_selections = node.gpu_selections.union(gpu_id)
        self.update_gpu_selections_of_parent(node.parent, gpu_id)

    def get_parent_gpu_selections(self, node: LPTreeNode):
        if not node:
            return set()
        if node.gpu_selections:
            return node.gpu_selections
        return self.get_parent_gpu_selections(node.parent)

    def insert_then_evict_from_runtime_cache(self, input_ids, runtime_selected):
        runtime_cache = self.runtime_caches[runtime_selected]
        node = runtime_cache.insert(tuple(input_ids))
        current_max_tokens = self.max_tokens_gpu[runtime_selected]
        if runtime_cache.evictable_size() > current_max_tokens:
            num_tokens = runtime_cache.evictable_size() - current_max_tokens
            runtime_cache.evict(num_tokens, lambda node: self.evict_callback(node, runtime_selected))
            print(f"GPU {runtime_selected} Evictable size: ", runtime_cache.evictable_size(), current_max_tokens)


    def evict_callback(self, node: LPTreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        updated_node = self.gpu_allocations.get(node)
        if updated_node:
            updated_node.remove(runtime_selected)
            if len(updated_node) == 0:
                del self.gpu_allocations[node]

        num_tokens = len(node.value)
        return len(node.value)

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
            is_small_node = leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens
            if is_small_node:
                gpu_selected = self.get_parent_gpu_selections(leaf_node.parent)
                assert len(gpu_selected) == 1
                runtime_idx = list(gpu_selected)[0]
                self.mem_cost[runtime_idx] += leaf_node.context_length
                self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
            else:
                cost_f = lambda gpu_id: leaf_node.context_length + self.mem_cost[gpu_id]
                gpu_selected = min(range(self.num_gpus), key=cost_f)
                gpu_selected = set([gpu_selected])
                runtime_idx = list(gpu_selected)[0]
                self.mem_cost[runtime_idx] += leaf_node.context_length
                self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
            self.counter += 1
            # self.insert_then_evict_from_runtime_cache(input_ids, runtime_idx)
            self.metrics_dict.append(
                {
                    "text": text[:100],
                    "rid": request_id,
                    "selected_runtime": runtime_idx,
                    "overhead": time.time() - start_time,
                    "mem_costs": self.mem_cost,
                    "parent_memory_cost": self.get_recomp_cost(leaf_node.parent, runtime_idx),
                    "current_leaf_node_cost": leaf_node.num_tokens,
                }
            )

            return int(runtime_idx)

    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
    ):
        with self.lock:
            pass
            # node = self.cache.find_node(input_ids)
            # self.mem_cost[func_output.runtime_selected] -= node.context_length

            # self.cache.remove_completed_input_ids(input_ids)
            # self.runtime_caches[func_output.runtime_selected].remove_completed_input_ids(input_ids)

    def print(self):
        _print_helper(self.cache.root_node, 0)

def _print_helper(node, indent, depth=0):
    for key, child in node.children.items():
        print(" " * indent, tokenizer.decode(child.value)[:20], child.gpu_selections, len(child.value))
        _print_helper(child, indent=indent + 2, depth=depth + 1)