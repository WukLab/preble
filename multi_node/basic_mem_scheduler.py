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


class BasicMemSchedulerV2:
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
            if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens:
                gpu_selected = self.get_parent_gpu_selections(leaf_node)
                for gpu in gpu_selected:
                    self.mem_cost[gpu] += self.get_recomp_cost(leaf_node, gpu)
            else:
                recom_costs = []
                for gpu_id in range(self.num_gpus):
                    recomputation_cost = self.get_recomp_cost(leaf_node, gpu_id)
                    recom_costs.append(recomputation_cost)
                gpu_selected = int(np.argmin([recom_costs[gpu_id] + self.mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
                self.mem_cost[gpu_selected] += recom_costs[gpu_selected]
                gpu_selected = set([gpu_selected])
            self.update_gpu_selections_of_parent(leaf_node, gpu_selected)

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

    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
    ):
        with self.lock:
            pass

    def print(self):
        _print_helper(self.cache.root_node, 0)


class BasicMemSchedulerLoadOnly:
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
            if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens:
                gpu_selected = self.get_parent_gpu_selections(leaf_node)
                for gpu in gpu_selected:
                    self.mem_cost[gpu] += 1
            else:
                recom_costs = []
                for gpu_id in range(self.num_gpus):
                    recomputation_cost = self.get_recomp_cost(leaf_node, gpu_id)
                    recom_costs.append(recomputation_cost)
                gpu_selected = int(np.argmin([self.mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
                self.mem_cost[gpu_selected] += 1
                gpu_selected = set([gpu_selected])
            self.update_gpu_selections_of_parent(leaf_node, gpu_selected)

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

    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
    ):
        pass
        # with self.lock:
        #     self.mem_cost[func_output.runtime_selected] -= self.cache.find_node(input_ids).context_length
    def print(self):
        _print_helper(self.cache.root_node, 0)
