from typing import Optional
from collections import defaultdict
import heapq
import time
from collections import defaultdict
from uuid import uuid4
import copy
import threading
import logging
from benchmarks.benchmark_utils import RequestFuncOutput
from histogram_based_scheduling import SlidingWindowHistogram
from datetime import datetime, timedelta

logging = logging.getLogger(__name__)
DEBUG_COUNTER = 0
class LpNode:
    def __init__(self, node_id, num_gpus):
        self.node_id = node_id
        self.variables = [
            None for _ in range(num_gpus)
        ]  # Will be initialized as binary variables in the model
        self.children_token_cost_at_max_depth = 0  # Issue is that depth_limit will cut off the tokens for children and that will treat it as free
        self.randomly_selected_gpu = None
        self.load_variables = [None for _ in range(num_gpus)]
        self.common_load = None

    def __repr__(self):
        variable_values = [var.x if var else None for var in self.variables]
        load_variable_values = [var.x if var else None for var in self.load_variables]
        common_load = self.common_load.x if self.common_load else None
        # ignore printing laod variables if None
        if any(load_variable_values):
            return f"LpNode(node_id={self.node_id}, variables={variable_values}, load_variables={load_variable_values}, common_load={common_load})"
        else:
            return f"LpNode(node_id={self.node_id}, variables={variable_values})"


class LPTreeNode:
    def __init__(self):
        self.id = uuid4()
        self.children = defaultdict(LPTreeNode)
        self.parent: Optional[LPTreeNode] = None
        self.value = None
        self.ref_counter = 0
        self.is_evicted = False
        self.load = 0
        self.last_access_time = time.time()
        self.gpu_selections = set()
        self.is_leaf = False
        self.decode_length = 0
        self.context_length = 0

    @property
    def num_tokens(self):
        return len(self.value)

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time

    def __eq__(self, other):
        if isinstance(other, LPTreeNode):
            return self.id == other.id  # Compare nodes based on their unique ID
        return False

    def __hash__(self):
        return hash(self.id)  # Use the unique ID for hashing

    def __repr__(self) -> str:
        return f"LPTreeNode(id={self.id}, ref_counter={self.ref_counter}, gpu_selections={self.gpu_selections})"


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class LPRadixCache:
    def __init__(self, histogram, disable=False):
        self.reset()
        self.disable = disable
        self.histogram = histogram

    ##### Public API #####

    def reset(self):
        self.root_node = LPTreeNode()
        self.root_node.value = []
        self.root_node.ref_counter = 1
        self.evictable_size_ = 0

    def find_node(self, key):
        if self.disable:
            return None
        current_gpu_selection, node = self.match_prefix_get_gpu_selection(key)
        return node

    def match_prefix_get_gpu_selection(self, key, path_to_node=[]):
        if self.disable:
            return [], self.root_node

        value = []
        current_gpu_selection = self.root_node.gpu_selections
        current_gpu_selection, node = self._match_prefix_helper_gpu_selection(
            self.root_node, key, value, current_gpu_selection
        )
        return current_gpu_selection, node

    def _match_prefix_helper_gpu_selection(
        self, node, key, value, current_gpu_selection
    ):
        child: LPTreeNode
        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                if child.gpu_selections:
                    current_gpu_selection = child.gpu_selections
                if prefix_len < len(c_key):
                    print(prefix_len, len(c_key))
                    assert False
                    new_node = self._split_node(
                        c_key, child, prefix_len, new_nodes_created=new_nodes_created
                    )
                    value.append(new_node.value)
                    # last_node[0] = new_node
                else:
                    value.append(child.value)
                    # last_node[0] = child
                    return self._match_prefix_helper_gpu_selection(
                        child, key[prefix_len:], value, current_gpu_selection
                    )
        return current_gpu_selection, node

    def match_prefix_return_str(self, key):
        return "".join(self.match_prefix(key)[0])

    def insert(
        self,
        key,
        value=None,
        node_map=None,
        all_modified_nodes=None,
        split_nodes=None,
        depth_limit=0,
    ):
        if node_map is None:
            node_map = {}
        if all_modified_nodes is None:
            all_modified_nodes = set()
        if split_nodes is None:
            split_nodes = {}  # key -> node
        if self.disable:
            return len(key)

        if value is None:
            value = [x for x in key]
        modified_nodes = set()
        created_node = self._insert_helper(
            self.root_node,
            key,
            value,
            node_map=node_map,
            modified_nodes=modified_nodes,
            depth_limit=depth_limit,
            current_depth=0,
            split_nodes=split_nodes,
        )

        node: LPTreeNode = created_node
        while node is not None:
            if node in all_modified_nodes:
                break
            all_modified_nodes.add(node)
            node = node.parent
        return created_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens, evict_callback):
        if self.disable:
            raise RuntimeError()

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.ref_counter > 0:
                continue

            num_evicted += evict_callback(x)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def smart_evict(self, num_tokens, evict_callback):
        if self.disable:
            raise RuntimeError()

        nodes = self.create_priority_queue()

        num_evicted = 0
        while num_evicted < num_tokens and len(nodes):
            x: LPTreeNode
            (cost, x) = heapq.heappop(nodes)

            if x == self.root_node:
                break

            if x.ref_counter > 0:
                continue

            num_evicted += evict_callback(x)
            self._delete_node(x)
            if len(x.parent.children) == 0:
                heapq.heappush(nodes, (x.parent.load * x.parent.num_tokens, x.parent))
    
    def inc_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.ref_counter += 1
            node = node.parent
        return delta

    def dec_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            # if node.ref_counter == 1: TODO why does this exist?
            #     self.evictable_size_ += len(node.value)
            #     delta += len(node.value)
            node.ref_counter -= 1
            node = node.parent
        return delta

    def remove_completed_input_ids(self, input_ids):
        node = self.find_node(input_ids)
        self.dec_ref_counter(node)  # remove reference counter up to parent
    
    def evictable_size(self):
        return self.evictable_size_

    def _split_node(
        self, key, child: LPTreeNode, split_len, node_map, depth_limit, current_depth
    ):
        # new_node -> child
        new_node = LPTreeNode()
        new_node.gpu_selections = copy.deepcopy(child.gpu_selections)
        new_node.children = {key[split_len:]: child}
        new_node.parent = child.parent
        new_node.ref_counter = child.ref_counter
        new_node.context_length = child.parent.context_length + split_len

        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.value = child.value[split_len:]

        new_node.parent.children[key[:split_len]] = new_node
        del new_node.parent.children[key]
        return new_node

    def _insert_helper(
        self,
        node: LPTreeNode,
        key,
        value,
        node_map,
        modified_nodes,
        depth_limit,
        current_depth,
        split_nodes,
        parent_context_length = 0
    ):
        node.last_access_time = time.time()
        node.ref_counter += 1
        node.is_evicted = False
        node.load += 1

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    child.ref_counter += 1
                    child.load += 1
                    child.is_evicted = False
                    modified_nodes.add(child)
                    return child
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return self._insert_helper(
                        child,
                        key,
                        value,
                        node_map=node_map,
                        modified_nodes=modified_nodes,
                        depth_limit=depth_limit,
                        current_depth=current_depth + 1,
                        split_nodes=split_nodes,
                        parent_context_length=parent_context_length + prefix_len,
                    )

            if prefix_len:
                new_node = self._split_node(
                    c_key,
                    child,
                    prefix_len,
                    node_map,
                    depth_limit=depth_limit,
                    current_depth=current_depth + 1,
                )
                # modified_nodes.add(new_node)
                # modified_nodes.add(child)
                # TODO check if this makes sense to ignore this?
                # if child in node_map and current_depth < depth_limit:
                split_nodes[child] = new_node
                return self._insert_helper(
                    new_node,
                    key[prefix_len:],
                    value[prefix_len:],
                    node_map=node_map,
                    modified_nodes=modified_nodes,
                    depth_limit=depth_limit,
                    current_depth=current_depth + 1,
                    split_nodes=split_nodes,
                    parent_context_length=parent_context_length + prefix_len,
                )

        if len(key):
            new_node = LPTreeNode()
            new_node.gpu_selections = set()
            new_node.parent = node
            new_node.value = value
            new_node.ref_counter = 1
            new_node.load = 1
            new_node.context_length = parent_context_length + len(key)

            node.children[key] = new_node
            self.evictable_size_ += len(value)
            # if current_depth < depth_limit:
            modified_nodes.add(new_node)
            # return new_node
            return new_node
        return node

    def _print_helper(self, node, indent, depth=0):
        if depth == 5:
            return
        for key, child in node.children.items():
            print(" " * indent, len(key), key[:10], f"r={child.ref_counter} {child.gpu_selections}")
            self._print_helper(child, indent=indent + 2, depth=depth + 1)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(k)
    
    def recursive_delete(self, cur_node, parent):
        if cur_node.children:
            for child_size, child_node in list(cur_node.children.items()):
                self.recursive_delete(child_node, cur_node)
        if parent is not None:
            children = list(parent.children.items())
            for key, value in children:
                value: LPTreeNode
                if value == cur_node:
                    value.is_evicted = True
                    self.evictable_size_ -= len(key)  # Adjust evictable size based on the node's size

    def _delete_node(self, cur_node):
        self.recursive_delete(cur_node, cur_node.parent)

    def _total_size_helper(self, node):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list

    def _collect_nodes(self):
        ret_list = []

        def dfs_(cur_node):
            ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list

    def create_priority_queue(self):
        nodes = self._collect_nodes()
        priority_queue = []
        for node in nodes:
            node: LPTreeNode
            if node.ref_counter == 0 and not node.is_evicted:
                assert node.load >= node.ref_counter
                priority = node.load * node.num_tokens # Min heap python
                heapq.heappush(priority_queue, (priority, node))
        return priority_queue

class MemSchedulerEvictBasedOnLoadHistogram:
    def __init__(self, num_nodes=2) -> None:
        self.mem_cost = [0 for _ in range(num_nodes)]
        self.gpu_allocations = defaultdict(set)
        self.num_gpus = num_nodes
        self.lock = threading.Lock()
        self.histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), num_buckets=10, num_gpus=2)
        self.cache = LPRadixCache(histogram=self.histogram)

        self.metrics_dict = []
        self.runtime_caches = [LPRadixCache(histogram=self.histogram) for _ in range(num_nodes)]
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
            runtime_cache.smart_evict(num_tokens, lambda node: self.evict_callback(node, runtime_selected))
            print(f"GPU {runtime_selected} Evictable size: ", runtime_cache.evictable_size(), current_max_tokens)


    def evict_callback(self, node: LPTreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        # TODO: Maybe update the parent if child has no parent
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
                recomputation_cost = self.get_recomp_cost(leaf_node, gpu_id)
                recom_costs.append(recomputation_cost)
            
            is_small_node = leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens
            cost_f = lambda gpu_id: recom_costs[gpu_id] + self.mem_cost[gpu_id]
            if is_small_node:
                gpu_selected = self.get_parent_gpu_selections(leaf_node.parent)
                if len(gpu_selected) != 1:
                    runtime_idx = min(list(gpu_selected), key=cost_f)
                else:
                    runtime_idx = list(gpu_selected)[0]
                self.mem_cost[runtime_idx] += recom_costs[runtime_idx]
                self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
            else:
                gpu_selected = min(range(self.num_gpus), key=cost_f)
                gpu_selected = set([gpu_selected])
                runtime_idx = list(gpu_selected)[0]
                self.mem_cost[runtime_idx] += recom_costs[runtime_idx]
                self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
            # self.mem_cost[runtime_idx] += recom_costs[runtime_idx]
            # Maybe memory cost should only be updated for the one that gets selected not scheduled
            self.counter += 1
            if self.counter % 100 == 0:
                print(self.mem_cost)
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
            self.cache.remove_completed_input_ids(input_ids)
            self.runtime_caches[func_output.runtime_selected].remove_completed_input_ids(input_ids)

class MemSchedulerEvictBasedOnLoadHistogramWithoutHeavyNodes:
    def __init__(self, num_nodes=2) -> None:
        self.mem_cost = [0 for _ in range(num_nodes)]
        self.gpu_allocations = defaultdict(set)
        self.num_gpus = num_nodes
        self.lock = threading.Lock()
        self.histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), num_buckets=10, num_gpus=2)
        self.cache = LPRadixCache(histogram=self.histogram)

        self.metrics_dict = []
        self.runtime_caches = [LPRadixCache(histogram=self.histogram) for _ in range(num_nodes)]
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
            runtime_cache.smart_evict(num_tokens, lambda node: self.evict_callback(node, runtime_selected))
            print(f"GPU {runtime_selected} Evictable size: ", runtime_cache.evictable_size(), current_max_tokens)


    def evict_callback(self, node: LPTreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        # TODO: Maybe update the parent if child has no parent
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
                recomputation_cost = self.get_recomp_cost(leaf_node, gpu_id)
                recom_costs.append(recomputation_cost)
            cost_f = lambda gpu_id: recom_costs[gpu_id] + self.mem_cost[gpu_id]
            gpu_selected = min(range(self.num_gpus), key=cost_f)
            gpu_selected = set([gpu_selected])
            runtime_idx = list(gpu_selected)[0]
            self.mem_cost[runtime_idx] += recom_costs[runtime_idx]
            self.update_gpu_selections_of_parent(leaf_node, {runtime_idx})
            # self.mem_cost[runtime_idx] += recom_costs[runtime_idx]
            # Maybe memory cost should only be updated for the one that gets selected not scheduled
            self.counter += 1
            self.insert_then_evict_from_runtime_cache(input_ids, runtime_idx)
            if self.counter % 100 == 0:
                print(self.mem_cost)
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
            self.cache.remove_completed_input_ids(input_ids)
            self.runtime_caches[func_output.runtime_selected].remove_completed_input_ids(input_ids)
