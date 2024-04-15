from gurobipy import GRB
import gurobipy as gp
from typing import Optional
from collections import defaultdict
import heapq
import time
from collections import defaultdict
from uuid import uuid4
import copy
import random
import threading
from enum import Enum, auto
import logging
from benchmarks.benchmark_utils import RequestFuncOutput
import os
import collections
import numpy as np

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
    def __init__(self, disable=False):
        self.reset()
        self.disable = disable

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
        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    child.ref_counter += 1
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
                        parent_context_length=node.context_length + len(child.value),
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
                    parent_context_length=node.context_length + len(child.value),
                )

        if len(key):
            new_node = LPTreeNode()
            new_node.gpu_selections = copy.deepcopy(node.gpu_selections)
            new_node.parent = node
            new_node.value = value
            new_node.ref_counter = 1
            node.children[key] = new_node
            node.context_length = parent_context_length + len(value) 
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
            print(" " * indent, len(key), key[:10], f"r={child.ref_counter}")
            self._print_helper(child, indent=indent + 2, depth=depth + 1)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(k)

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

class LPGurobiGreedyTraversal:
    def __init__(self, num_gpus, gpu_configs=None):
        self.lp_forward_simulation = gpu_configs[0].lp_forward_simulation if gpu_configs else None
        if self.lp_forward_simulation is None:
            self.lp_forward_simulation = lambda num_extend_tokens, is_leaf: num_extend_tokens * 0.148 + (22.7 if is_leaf else 0)

        self.num_gpus = num_gpus
        self.node_to_gpu_selections = defaultdict(set)
        self.depth_limit = 3
        self.current_load_cost = [0 for _ in range(num_gpus)]
        self.current_memory_cost = [0 for _ in range(num_gpus)]
        self.model = gp.Model()
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("LogToConsole", 0)
        self.model.setParam('Seed', 0)
        self.model.setParam("Threads", 0)
        # self.model.setParam("WorkLimit", 0.005)
        # self.model.setParam("MIPGap", 0.02)
        self.model.setParam("Method", 4)
        # self.model.setParam("TimeLimit", 0.005)

        self.variables_initialized = False
        self.initialize_or_update_variables()
        # self.init_cache() # for performance reasons, Presolve an LP. This reduces the cost of the first LP

    def initialize_or_update_variables(self):
        if not self.variables_initialized:
            # First run: add variables
            self.max_per_gpu_cost = self.model.addVar(name="max_per_gpu_cost", vtype=GRB.CONTINUOUS)
            self.lp_node = LpNode("main", self.num_gpus)
            for gpu in range(self.num_gpus):
                self.lp_node.variables[gpu] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"x_{gpu}"
                )
            self.variables_initialized = True
        else:
            # Subsequent runs: reset the model but keep the same structure
            self.model.reset()


    def _calculate_children_token_cost(self, node: LPTreeNode):
        """
        Recursively calculate the total number of tokens for all children of a given node,
        effectively aggregating the tokens for nodes that are beyond the depth limit.
        """
        if node is None:
            return 0
        total_tokens = node.num_tokens
        for child in node.children.values():
            total_tokens += self._calculate_children_token_cost(child)
        return total_tokens

    def update_constraints(self, leaf_node: LPTreeNode, modified_nodes: set[LPTreeNode], decode_cost):
        self.model.remove(self.model.getConstrs())
        self.model.addConstr(gp.quicksum(self.lp_node.variables) >= 1, "min_one_gpu")

        total_cost = gp.LinExpr()
        per_gpu_recomp_cost = [gp.LinExpr() for _ in range(self.num_gpus)]
        per_gpu_load_cost = [gp.LinExpr() for _ in range(self.num_gpus)]
        per_gpu_mem_load_cost = [gp.LinExpr() for _ in range(self.num_gpus)]
        new_total_memory_cost = [0 for _ in range(self.num_gpus)]

        decoding_time = decode_cost
        # print("Decoding time", decoding_time)
        node: LPTreeNode = leaf_node
        for gpu_index, var in enumerate(self.lp_node.variables):
            node = leaf_node
            total_recomputation_tokens = 0
            while node != None:
                if gpu_index in node.gpu_selections:
                    break
                total_recomputation_tokens += node.num_tokens
                node = node.parent

            recomputation_time = 0
            if total_recomputation_tokens != 0:
                recomputation_time = self.lp_forward_simulation(total_recomputation_tokens, leaf_node.context_length) * 1000
            total_cost += var * recomputation_time
            new_total_memory_cost[gpu_index] = recomputation_time
            per_gpu_recomp_cost[gpu_index] += var * recomputation_time
    
        for gpu_index in range(self.num_gpus):
            per_gpu_load_cost[gpu_index] += self.lp_node.variables[gpu_index] * decoding_time
            per_gpu_mem_load_cost[gpu_index] += self.current_memory_cost[gpu_index]  # Assuming current_memory_cost is tracked
            per_gpu_load_cost[gpu_index] += self.current_load_cost[gpu_index]

            cost =  self.current_load_cost[gpu_index] + self.current_memory_cost[gpu_index] + decoding_time
            # logging.info(f"Cost for GPU {gpu_index}: {cost}, load cost: {self.current_load_cost[gpu_index]}, memory cost: {self.current_memory_cost[gpu_index]}, decoding time: {decoding_time}")
            self.model.addConstr(
                per_gpu_mem_load_cost[gpu_index] + per_gpu_load_cost[gpu_index] + per_gpu_recomp_cost[gpu_index] <= self.max_per_gpu_cost,
                name=f"max_per_gpu_cost_constr_{gpu_index}"
            )

        self.model.setObjective(self.max_per_gpu_cost, GRB.MINIMIZE)
        return new_total_memory_cost, decoding_time

    def traverse_and_optimize(
        self, leaf_node: LPTreeNode, modified_nodes: set[LPTreeNode] = None, split_nodes={}, decode_cost=0
    ):
        start_time = time.time()
        leaf_node.is_leaf = True

        self.initialize_or_update_variables()
        for key, value in split_nodes.items():
            self.node_to_gpu_selections[value] = self.node_to_gpu_selections[key]

        new_total_memory_cost, total_decode_time = self.update_constraints(leaf_node, modified_nodes, decode_cost)
        self.model.optimize()
        global DEBUG_COUNTER
        DEBUG_COUNTER += 1
        self.model.write(f"logs/gurobiv2_{DEBUG_COUNTER}_{time.time()}.lp")
        # print(self.max_per_gpu_cost.X)
        if self.model.Status == GRB.OPTIMAL:
            pass
        elif self.model.Status == GRB.INFEASIBLE:
            print("Infeasable solution found")
        else:
            pass

        selected_gpus = self.update_gpu_selections(self.lp_node, leaf_node)
        # print(f"Selected GPUs: {selected_gpus}")
        for gpu in selected_gpus:
            self.current_load_cost[gpu] += total_decode_time
            self.current_memory_cost[gpu] += new_total_memory_cost[gpu]
        return time.time() - start_time


    def update_gpu_selections(self, lp_node, leaf_node):
        selected_gpus = [
            gpu_id for gpu_id, var in enumerate(self.lp_node.variables) if var.X >= 0.99
        ]
        leaf_node.gpu_selections = set(selected_gpus)
        self.node_to_gpu_selections[leaf_node] = leaf_node.gpu_selections

        node: LPTreeNode = leaf_node.parent
        while node != None:
            parent_gpu_selection = set()
            for key, children in node.children.items():
                parent_gpu_selection.update(children.gpu_selections)
            node.gpu_selections = parent_gpu_selection
            self.node_to_gpu_selections[node] = parent_gpu_selection.union(self.node_to_gpu_selections.get(node, set()))
            node = node.parent
        return selected_gpus
        
    def pretty_print(self, prefix_node, depth_limit=4, tokenizer=None):
        self.pretty_print_helper(
            prefix_node, depth_limit=depth_limit, tokenizer=tokenizer
        )

    def pretty_print_helper(
        self, prefix_node: LPTreeNode, indent="", depth=0, depth_limit=4, tokenizer=None
    ):
        if depth == depth_limit:
            return
        selected_gpus = self.node_to_gpu_selections.get(prefix_node)

        def get_tool(workload_item):
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            text = tokenizer.decode(workload_item)
            if ":" in text:
                return text.split(":")[0].strip().replace("\n", " ")
            else:
                return text[:16].strip().replace("\n", "")

        print(
            f"{indent}Node {prefix_node.id} (Tokens: {get_tool(prefix_node.value)}, {len(prefix_node.value)}, {(prefix_node.ref_counter)}): GPUs {selected_gpus}"
        )

        for child in prefix_node.children.values():
            self.pretty_print_helper(
                child,
                indent + "  ",
                depth=depth + 1,
                depth_limit=depth_limit,
                tokenizer=tokenizer,
            )

    def update_nodes_with_solution(self, modified_nodes=None):
        for prefix_node, lp_node in self.node_to_gpu_selections.items():
            prefix_node.gpu_selections = set()
            for gpu_id, var in enumerate(lp_node.variables):
                if var.X >= 0.99:
                    prefix_node.gpu_selections.add(gpu_id)

    def completed_request(self, tree_cache, input_ids, decode_cost):
        # decoding_time = lambda x: 47 * x
        total_decode_time = decode_cost
        node: LPTreeNode = tree_cache.find_node(input_ids)
        tree_cache.remove_completed_input_ids(input_ids)
        # for selection in node.gpu_selections:
            # self.current_load_cost[selection] -= total_decode_time

    def insert_into_cache_and_solve(self, input_ids, tree_cache, decode_cost):
        node_map = self.node_to_gpu_selections
        split_nodes = {}
        modified_nodes = set()
        node = tree_cache.insert(
            tuple(input_ids),
            node_map=node_map,
            all_modified_nodes=modified_nodes,
            depth_limit=self.depth_limit,
            split_nodes=split_nodes,
        )
        self.traverse_and_optimize(
            node, modified_nodes=modified_nodes, split_nodes=split_nodes, decode_cost=decode_cost
        )
        return node

class GurobiGreedyLPScheduler:
    class RuntimeSelectionType(Enum):
        RANDOM = auto()
        NOT_RANDOM = auto()

    def __init__(self, num_nodes: int, gpu_configs = None):
        self.num_nodes = num_nodes
        self.tree_cache = LPRadixCache()
        self.lp_tree_traversal = LPGurobiGreedyTraversal(num_nodes, gpu_configs)
        self.lp_tree_traversal.depth_limit = 64
        self.metrics_dict = []
        self.load = {}
        self.lock = threading.Lock()
        self.gpu_configs = gpu_configs

        self.runtime_caches = [LPRadixCache() for _ in range(num_nodes)]
        self.max_tokens_gpu = [198466, 198466]
        self.average_tpot = 0.041
        self.counter = 0
        self.rid_to_deocde_cost = {}
        self.tpot_queue = collections.deque(maxlen=100)
        global DEBUG_COUNTER
        DEBUG_COUNTER = 0
        # self.tpot_queue.append(0.04)
        # self.tpot_queue.append(0.041)
        # self.tpot_queue.append(0.041)
        # self.tpot_queue.append(0.041)

    def evict_callback(self, node: LPTreeNode, runtime_selected: int):
        """Method to handle eviction logic."""
        updated_node = self.lp_tree_traversal.node_to_gpu_selections.get(node)
        if updated_node:
            updated_node.remove(runtime_selected)
            if len(updated_node) == 0:
                del self.lp_tree_traversal.node_to_gpu_selections[node]
        num_tokens = len(node.value)
        # accumlation
        mistral_tokens_to_prefill_time = self.lp_tree_traversal.lp_forward_simulation(num_tokens, node.context_length) * 1000
        # mistral_tokens_to_prefill_time = 0.148 * num_tokens
        # if node.is_leaf:
        #     mistral_tokens_to_prefill_time += 22.7

        # TODO DO I Need to evict here?
        # print(f"Eviction cost: mistral_tokens_to_prefill_time: {mistral_tokens_to_prefill_time}, num_tokens: {num_tokens}")
        self.lp_tree_traversal.current_memory_cost[runtime_selected] -= mistral_tokens_to_prefill_time
        return len(node.value)

    def select_runtime_from_gpu_selections(self, gpu_selections) -> tuple[int, RuntimeSelectionType]:
        mode = GurobiGreedyLPScheduler.RuntimeSelectionType.NOT_RANDOM
        if len(gpu_selections) == 0 or len(gpu_selections) == self.num_nodes:
            gpu_selections = set(range(self.num_nodes))
            mode = GurobiGreedyLPScheduler.RuntimeSelectionType.RANDOM
        runtime_selected = random.choice(list(gpu_selections))
        return runtime_selected, mode

    def insert_then_evict_from_runtime_cache(self, input_ids, runtime_selected):
        runtime_cache = self.runtime_caches[runtime_selected]
        node = runtime_cache.insert(tuple(input_ids))
        current_max_tokens = self.max_tokens_gpu[runtime_selected]
        if runtime_cache.evictable_size() > current_max_tokens:
            num_tokens = runtime_cache.evictable_size() - current_max_tokens
            runtime_cache.evict(num_tokens, lambda node: self.evict_callback(node, runtime_selected))
            # print(f"GPU {runtime_selected} Evictable size: ", runtime_cache.evictable_size(), current_max_tokens)

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
            print(f"Request ID: {request_id}, Text: {text[-50:]}")
            st = time.time()
            # p_90_tpot = 6.7 / 1000
            # decode_cost =  sampling_params.get("max_new_tokens") * tpot_queue_ms
            p_90 = np.percentile(self.tpot_queue if self.tpot_queue else [0.0], 50)
            print(f"TPOT: {p_90}, Max new tokens: {sampling_params.get('max_new_tokens')}, Decode cost: {p_90 * sampling_params.get('max_new_tokens') / 1000}")
            # decode_cost = p_90_tpot * sampling_params.get("max_new_tokens") / 1000 # 1000 bc converting seconds to miliseconds
            # 0.0004393649647164497
            # 0.00067
            decode_cost = max(p_90, 0.0001) * sampling_params.get("max_new_tokens")
            self.rid_to_deocde_cost[request_id] = decode_cost 
            # request_id -> memory_cost
            node = self.lp_tree_traversal.insert_into_cache_and_solve(input_ids, self.tree_cache, decode_cost)
            solving_time = time.time() - st

            gpu_selections: set[int] = node.gpu_selections
            runtime_selected, selection_type = self.select_runtime_from_gpu_selections(gpu_selections)
            self.load[runtime_selected] = self.load.get(runtime_selected, 0) + 1
            self.insert_then_evict_from_runtime_cache(input_ids, runtime_selected)
        if time.time() - start_time > 0.03:
            print(f"Overall time", time.time() - start_time, solving_time)
        self.metrics_dict.append(
            {
                "text": text,
                "rid": request_id,
                "selected_runtime": runtime_selected,
                "overhead": time.time() - start_time,
                "mode": selection_type,
            }
        )
        return runtime_selected

    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output: RequestFuncOutput=None
    ):
        with self.lock:
            self.tpot_queue.append(func_output.tpot)
            if request_id not in self.rid_to_deocde_cost:
                decode_cost = 0
                assert False
            else:
                decode_cost = self.rid_to_deocde_cost.pop(request_id)
            self.lp_tree_traversal.completed_request(self.tree_cache, input_ids, decode_cost)
            self.runtime_caches[func_output.runtime_selected].remove_completed_input_ids(input_ids)
            # tpot = func_output.tpot
            # self.average_tpot = (self.average_tpot * self.counter + tpot) / (self.counter + 1)
            self.counter += 1

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

if __name__ == "__main__":
    pass
    # import random
    # from transformers import AutoTokenizer
    # import sys
    # import os
    # import copy
    # import random

    # # Add the parent directory of the 'src' directory to the Python path
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname("."), "..")))
    # from transformers import AutoTokenizer
    # from benchmarks.benchmark_workload_gen import ToolBenchDataLoader, LoadDistribution

    # cache = LPRadixCache()

    # num_workloads = 100
    # num_requests = 4096
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # random.seed(5)
    # dataloader = ToolBenchDataLoader(
    #     "benchmarks/datasets/G1_workload_updated_input_output_lengths_4096_cropped_to_50.json",
    #     num_workloads,
    #     num_requests,
    #     tokenizer,
    #     LoadDistribution.ZIPF,
    # )
    # workload = dataloader.generate_workload(k=1.1)

    # scheduler = GurobiGreedyLPScheduler(2)
    # for i, item in enumerate(workload[:64]):
    #     runtime_selected = scheduler.runtime_selector(
    #         text=item["text"], request_id=i, input_ids=item["input_ids"]
    #     )
    #     # print(item["text"], runtime_selected)
    # # print(pd.DataFrame(scheduler.metrics_dict))
    # scheduler.lp_tree_traversal.pretty_print(
    #     scheduler.tree_cache.root_node, depth_limit=3, tokenizer=tokenizer
    # )
    # breakpoint()
    # scheduler.lp_tree_traversal.pretty_print(scheduler.tree_cache.root_node, depth_limit=3)


# bursty load -> tool replicated on every node. Only tool 1 - 50. 
# non bursty load. tool 1 - 50 

# Toolbench bursty 
# Loogle <- cache of tools is empty
# toolbench 