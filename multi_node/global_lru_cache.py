from typing import Optional
from collections import defaultdict
import heapq
import time
from collections import defaultdict
from uuid import uuid4
import copy
import threading
import numpy as np
import logging
from benchmarks.benchmark_utils import RequestFuncOutput
from datetime import datetime, timedelta
from typing import List, Tuple

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
    def __init__(self, num_nodes=2):
        self.id = uuid4()
        self.children = defaultdict(LPTreeNode)
        self.parent: Optional[LPTreeNode] = None
        self.value = None
        self.ref_counter = [0 for _ in range(num_nodes)]
        self.load = 0
        self.last_access_time = time.time()
        self.evicted_gpus = set()
        self.cached_gpus = set()
        self.is_leaf = False
        self.decode_length = 0
        self.context_length = 0
        self.decoding_tree_node: LPTreeNode = None

    def has_cached_gpu(self, gpu):
        return gpu in self.cached_gpus and gpu not in self.evicted_gpus

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
        return f"LPTreeNode(id={self.id}, ref_counter={self.ref_counter}, cached_gpus={self.cached_gpus}, evicted_gpus:{self.evicted_gpus})"

class SlidingWindowHistogram:
    def __init__(self, window_duration, num_gpus=2):
        self.window_duration = window_duration
        self.histogram = defaultdict(int)
        self.timestamps: List[Tuple[datetime, LPTreeNode]] = []
        self.num_gpus = num_gpus

    def update(self, node: LPTreeNode):
        timestamp = datetime.now()
        self.timestamps.append((timestamp, node))
        self.histogram[node] += 1
        self._remove_old_entries(timestamp)

    def _remove_old_entries(self, current_timestamp):
        window_start = current_timestamp - self.window_duration
        while self.timestamps and self.timestamps[0][0] < window_start:
            timestamp, node = self.timestamps.pop(0)
            self.histogram[node] -= 1
            if self.histogram[node] == 0: # Remove the gpu selections from it if there's no load
                node.cached_gpus = set() # Reset the gpu selections for older entries
            if self.histogram[node] <= 0:
                del self.histogram[node]
            
    
    def rename_node(self, old_node, new_node):
        if old_node in self.histogram:
            self.histogram[new_node] = self.histogram.get(old_node)
            rename_mapping = {old_node: new_node}
            self.timestamps = [(timestamp, rename_mapping.get(node, node)) for timestamp, node in self.timestamps]
    
    def copy_node(self, old_node, new_node):
        if old_node not in self.histogram:
            return 
        self.histogram[new_node] = self.histogram.get(old_node) # the counts of both should be the same
        new_timestamps = []
        for timestamp, node in self.timestamps:
            if node == old_node:
                new_timestamps.append((timestamp, new_node))
                new_timestamps.append((timestamp, old_node))
        self.timestamps = new_timestamps

    def query(self):
        return dict(self.histogram)

    def get(self, node):
        return self.histogram.get(node, 0)

    def current_allocation_per_gpu(self):
        allocation = [0 for _ in range(self.num_gpus)]
        node: LPTreeNode
        for node, cost in self.histogram.items():
            for gpu in node.cached_gpus:
                allocation[gpu] += cost # potentionally divide by length of node.cached_gpus here
        return allocation


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class LPRadixCache:
    def __init__(self, histogram, disable=False, num_gpus=2):
        self.num_gpus = num_gpus
        self.reset()
        self.disable = disable
        self.histogram: SlidingWindowHistogram = histogram

    ##### Public API #####

    def reset(self):
        self.root_node = LPTreeNode(num_nodes=self.num_gpus)
        self.root_node.value = []
        self.root_node.ref_counter = [1 for _ in range(self.num_gpus)]
        self.allocated_size_ = [0 for _ in range(self.num_gpus)]

    def find_node(self, key):
        if self.disable:
            return None
        current_gpu_selection, node = self.match_prefix_get_gpu_selection(key)
        return node

    def match_prefix_get_gpu_selection(self, key, path_to_node=[]):
        if self.disable:
            return [], self.root_node

        value = []
        current_gpu_selection = self.root_node.cached_gpus
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
                if child.cached_gpus:
                    current_gpu_selection = child.cached_gpus
                if prefix_len < len(c_key):
                    print(prefix_len, len(c_key))
                    # assert False
                    return {}, None
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
        # for old_node, new_node in split_nodes.items():
        #     self.histogram.copy_node(old_node=old_node, new_node=new_node)

        # node: LPTreeNode = created_node
        # while node is not None:
        #     if node in all_modified_nodes:
        #         break
        #     self.histogram.update(node)
        #     all_modified_nodes.add(node)
        #     node = node.parent
        
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

    def evict_with_runtime_id_without_removing(self, num_tokens, evict_callback, runtime_id):
        if self.disable:
            raise RuntimeError()

        leaves = self.collected_nodes_with_runtime_idx(runtime_id)
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)
            x: LPTreeNode
            if x == self.root_node:
                break
            if x.ref_counter[runtime_id] > 0:
                continue
            if x.has_cached_gpu(runtime_id):
                self.allocated_size_[runtime_id] -= len(x.value)
            num_evicted += evict_callback(x)
            # self._delete_leaf(x)
            # self._delete_node(x, runtime_id)
            # if len(x.parent.children) == 0:
            #     heapq.heappush(leaves, x.parent)

    def dec_ref_counter(self, node, runtime_id):
        delta = 0
        node:LPTreeNode
        while node != self.root_node:
            node.ref_counter[runtime_id] -= 1
            assert node.ref_counter[runtime_id] >= 0
            node = node.parent
        return delta

    def remove_completed_input_ids(self, input_ids, runtime_id):
        node = self.find_node(input_ids)
        self.dec_ref_counter(node, runtime_id)  # remove reference counter up to parent
    
    def allocated_size(self, runtime_id):
        return self.allocated_size_[runtime_id]

    def _split_node(
        self, key, child: LPTreeNode, split_len, node_map, depth_limit, current_depth
    ):
        # new_node -> child
        new_node = LPTreeNode(num_nodes=self.num_gpus)
        new_node.cached_gpus = copy.deepcopy(child.cached_gpus)
        new_node.evicted_gpus = copy.deepcopy(child.evicted_gpus)
        new_node.children = {key[split_len:]: child}
        new_node.parent = child.parent
        new_node.ref_counter = copy.deepcopy(child.ref_counter)
        new_node.load = child.load

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
        node.load += 1

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    child.load += 1
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
            new_node.cached_gpus = set()
            new_node.evicted_gpus = set()
            new_node.parent = node
            new_node.value = value
            new_node.load = 1
            new_node.context_length = parent_context_length + len(key)

            node.children[key] = new_node
            # self.allocated_size_ += len(value)
            # if current_depth < depth_limit:
            modified_nodes.add(new_node)
            # return new_node
            return new_node
        return node

    def update_allocated_size(self, node: LPTreeNode, runtime_id):
        # handle decoding tokens
        while node:
            node.ref_counter[runtime_id] += 1
            if not node.has_cached_gpu(runtime_id):
                self.allocated_size_[runtime_id] += len(node.value)
                if runtime_id in node.evicted_gpus:
                    node.evicted_gpus.remove(runtime_id)
                node.cached_gpus.add(runtime_id)
            node = node.parent

    def _print_helper(self, node, indent, depth=0):
        if depth == 5:
            return
        for key, child in node.children.items():
            print(" " * indent, len(key), key[:10], f"r={child.ref_counter} {child.cached_gpus}")
            self._print_helper(child, indent=indent + 2, depth=depth + 1)

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

        def dfs_(cur_node: LPTreeNode):
            ret_list.append(cur_node)
            if cur_node.decoding_tree_node:
                ret_list.append(cur_node.decoding_tree_node)
            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list

    def create_priority_queue(self, runtime_id):
        nodes = self._collect_nodes()
        priority_queue = []
        for node in nodes:
            node: LPTreeNode
            if node.ref_counter[runtime_id] == 0 and runtime_id in node.cached_gpus:
                # load = self.histogram.get(node) / len(node.cached_gpus)
                # assert load >= node.ref_counter
                # priority = load * node.num_tokens # Min heap python
                heapq.heappush(priority_queue, (node.last_access_time, node))
        return priority_queue

    def collected_nodes_with_runtime_idx(self, runtime_id):
        nodes = self._collect_nodes()
        priority_queue = []
        current_allocated_size = 0
        for node in nodes:
            if node.has_cached_gpu(runtime_id):
                current_allocated_size += len(node.value)
            node: LPTreeNode
            if node.ref_counter[runtime_id] == 0 and node.has_cached_gpu(runtime_id):
                heapq.heappush(priority_queue, node)

        self.allocated_size_[runtime_id] = current_allocated_size
        return priority_queue

    def get_evictable_size(self, runtime_id):
        nodes = self._collect_nodes()
        current_allocated_size = 0
        for node in nodes:
            node: LPTreeNode
            if node.ref_counter[runtime_id] == 0 and node.has_cached_gpu(runtime_id):
                current_allocated_size += len(node.value)
        return current_allocated_size