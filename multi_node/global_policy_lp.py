# %%
import random

# Add the parent directory of the 'src' directory to the Python path
# %%
import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple
from data_parallel_request_cache import CustomRuntimeSelector
from uuid import uuid4
import torch

class LPTreeNode:
    def __init__(self):
        self.id = uuid4()  
        self.children = defaultdict(LPTreeNode)
        self.parent = None
        self.value = None
        self.ref_counter = 0
        self.last_access_time = time.time()
        self.gpu_selections = set()

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

    def match_prefix_get_gpu_selection(self, key):
        if self.disable:
            return [], self.root_node

        value = []
        current_gpu_selection = self.root_node.gpu_selections
        current_gpu_selection = self._match_prefix_helper_gpu_selection(self.root_node, key, value, current_gpu_selection)
        return current_gpu_selection

    def _match_prefix_helper_gpu_selection(self, node, key, value, current_gpu_selection):
        node.last_access_time = time.time()
        child: LPTreeNode
        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                text = tokenizer.decode(child.value)
                if child.gpu_selections:
                    current_gpu_selection = child.gpu_selections
                if prefix_len < len(c_key):
                    assert False
                    new_node = self._split_node(c_key, child, prefix_len, new_nodes_created=new_nodes_created)
                    value.append(new_node.value)
                    # last_node[0] = new_node
                else:
                    value.append(child.value)
                    # last_node[0] = child
                    return self._match_prefix_helper_gpu_selection(child, key[prefix_len:], value, current_gpu_selection)
        return current_gpu_selection

    def match_prefix_return_str(self, key):
        return "".join(self.match_prefix(key)[0])

    def insert(self, key, value=None, node_map=None):
        if node_map is None:
            node_map = {}
        if self.disable:
            return len(key)

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, node_map=node_map)

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

            num_evicted += evict_callback(x.value)
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
            if node.ref_counter == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.ref_counter -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def _split_node(self, key, child, split_len, node_map):
        # new_node -> child
        is_initial_code_in_node_map = child in node_map
        if is_initial_code_in_node_map:
            lp_node = node_map[child]

        new_node = LPTreeNode()
        new_node.gpu_selections = copy.deepcopy(child.gpu_selections)
        new_node.children = {
            key[split_len:]: child
        }
        new_node.parent = child.parent
        new_node.ref_counter = child.ref_counter

        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.value = child.value[split_len:]

        new_node.parent.children[key[:split_len]] = new_node
        del new_node.parent.children[key]

        if is_initial_code_in_node_map:
            # NODE Map is updated because splitting the nodes needs to preserve solver state
            node_map[child] = lp_node
            node_map[new_node] = lp_node
            # Potentionally pop the previous entry

        if is_initial_code_in_node_map:
            assert child in node_map
            assert new_node in node_map
        return new_node

    def _insert_helper(self, node, key, value, node_map):
        node.last_access_time = time.time()
        node.ref_counter += 1
        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)

            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    child.ref_counter += 1
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value, node_map=node_map)

            if prefix_len:
                new_node = self._split_node(c_key, child, prefix_len, node_map)
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:], node_map=node_map
                )

        if len(key):
            new_node = LPTreeNode()
            new_node.gpu_selections = copy.deepcopy(node.gpu_selections)
            new_node.parent = node
            new_node.value = value
            new_node.ref_counter = 1
            node.children[key] = new_node
            self.evictable_size_ += len(value)
        return 0

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


# %%
import time
from mip import Model, xsum, BINARY, MINIMIZE, OptimizationStatus, minimize, INTEGER, GUROBI
from gurobipy import GRB, Env
from typing import Dict
# Initialize the Gurobi environment with output turned off
env = Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()

class LpNode:
    def __init__(self, node_id, num_gpus):
        self.node_id = node_id
        self.variables = [None] * num_gpus  # Will be initialized as binary variables in the model
        self.children_token_cost_at_max_depth = 0 # Issue is that depth_limit will cut off the tokens for children and that will treat it as free
    
class LPTreeTraversal:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.node_map = {}
        self.depth_limit = 5
        self.model = Model(sense=MINIMIZE, solver_name=GUROBI)

    def _traverse_tree(self, current_prefix_node:LPTreeNode, parent_lp_node: LpNode=None, depth=0):
        if depth == self.depth_limit:
            assert parent_lp_node is not None
            parent_lp_node.children_token_cost_at_max_depth = self._calculate_children_token_cost(current_prefix_node)
            return
        self.counter += 1
        current_lp_node = LpNode(self.counter, self.num_gpus)
        self.node_map[current_prefix_node] = current_lp_node
        # Initialize binary variables for the LP node
        for gpu in range(self.num_gpus):
            current_lp_node.variables[gpu] = self.model.add_var(f"node_{self.counter}_{gpu}",var_type=BINARY)

        # At least one GPU must be allocated for a prefix
        self.model += xsum(current_lp_node.variables) >= 1

        if parent_lp_node:
            # If the child takes a node, then the parent must also take a node
            for gpu in range(self.num_gpus):
                self.model += current_lp_node.variables[gpu] <= parent_lp_node.variables[gpu]

        for child_prefix_node in current_prefix_node.children.values():
            self._traverse_tree(current_prefix_node=child_prefix_node, parent_lp_node=current_lp_node, depth=depth + 1,)

    def _calculate_children_token_cost(self, node):
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

    def add_parent_child_gpu_constraints(self):
        for parent_prefix_node, parent_lp_node in self.node_map.items():
            if not parent_prefix_node.children:  # Skip leaf nodes
                continue
            for gpu_index in range(self.num_gpus):
                children_gpu_selections = []
                for child_prefix_node in parent_prefix_node.children.values():
                    if child_prefix_node in self.node_map:
                        child_lp_node = self.node_map[child_prefix_node]
                        children_gpu_selections.append(child_lp_node.variables[gpu_index])
                if children_gpu_selections:
                    children_selections_total = xsum(children_gpu_selections)
                    self.model += parent_lp_node.variables[gpu_index] <= children_selections_total

    def traverse_and_optimize(self, prefix_tree_root, existing_cost={}):
        start_time = time.time()

        # self.model.reset()  # Re-initialize the model for a new optimization problem
        self.model = Model(sense=MINIMIZE, solver_name=GUROBI)
        self.model.verbose = 0

        self.node_map = {}
        self.counter = 0

        self._traverse_tree(prefix_tree_root)  # Set up variables and base constraints
        self.add_parent_child_gpu_constraints()  # Add parent-child constraints

        # Objective components: Let's assume we're trying to minimize the total cost adjusted for existing costs
        total_cost = []
        per_gpu_cost = [[] for _ in range(self.num_gpus)]
        initial_solution = []
        total_cost_saved = 0
        for prefix_node, lp_node in self.node_map.items():
            node_costs = existing_cost.get(prefix_node, {})
            # children token cost is to account for depth cutoff
            num_tokens_total = prefix_node.num_tokens + lp_node.children_token_cost_at_max_depth
            for gpu_index, var in enumerate(lp_node.variables):
                previous_gpu_selected = existing_cost.get(prefix_node, {}).get(gpu_index, 0) 
                if previous_gpu_selected:
                    initial_solution.append((var, previous_gpu_selected))
                    total_cost_saved += previous_gpu_selected * num_tokens_total

                total_cost.append(var * num_tokens_total - var * previous_gpu_selected * num_tokens_total)
                per_gpu_cost[gpu_index].append(var * num_tokens_total)

        max_per_gpu_cost = self.model.add_var(name='per_gpu_cost_lim', var_type=INTEGER)
        for i in range(self.num_gpus):
            self.model += xsum(per_gpu_cost[i]) <= max_per_gpu_cost
        total_cost_var = self.model.add_var(name='total_cost', var_type=INTEGER)
        self.model += xsum(total_cost) == total_cost_var
    
        self.model.start = initial_solution
        self.model.threads = -1
        self.model.max_mip_gap = 0.01
        setup_time = time.time() - start_time
        start_time = time.time()
        self.model.objective = minimize(total_cost_var + max_per_gpu_cost)
        status = self.model.optimize()
        print(f"Solving time: {time.time() - start_time}s Setup Time {setup_time}s")

    def get_exisiting_cost(self):
        existing_cost = {}
        for prefix_node, lp_node in self.node_map.items():
            for gpu_id, var in enumerate(lp_node.variables):
                if prefix_node not in existing_cost:
                    existing_cost[prefix_node] = {}
                if var.x >= 0.99:
                    existing_cost[prefix_node][gpu_id] = 1
                else:
                    existing_cost[prefix_node][gpu_id] = 0
        return existing_cost

    def calculate_tokens_per_gpu(self):
        tokens_per_gpu = {gpu: 0 for gpu in range(self.num_gpus)}  # Reset/initialize
        load_to_gpu = {gpu: 0 for gpu in range(self.num_gpus)}
        for prefix_node, lp_node in self.node_map.items():
            for i, var in enumerate(lp_node.variables):
                if var.x >= 0.99:  # If GPU i is selected by this node, using .x for variable value in MIP
                    tokens_per_gpu[i] += prefix_node.num_tokens  # Accumulate tokens
                    load_to_gpu[i] += prefix_node.ref_counter
        return tokens_per_gpu, load_to_gpu


    def pretty_print(self, prefix_node):
        # This method will call pretty_print_helper and then print additional information
        # Adjustments are mainly in handling variable values using .x in MIP
        self.pretty_print_helper(prefix_node)
        tokens_per_gpu, load_to_gpu = self.calculate_tokens_per_gpu()
        print(f"Tokens per GPU: {tokens_per_gpu} {load_to_gpu}")
        print(f"Objective value: {self.model.objective_value}")

    def pretty_print_helper(self, prefix_node, indent="", depth=0):
        if depth == self.depth_limit:
            return
        lp_node = self.node_map.get(prefix_node)
        if lp_node:
            selected_gpus = [i for i, var in enumerate(lp_node.variables) if var.x >= 0.99]  # Adjust threshold as needed, using .x for variable value
            # if lp_node.node_id == 4 or True:
            print(f"{indent}Node {lp_node.node_id} (Tokens: {len(prefix_node.value)}): GPUs {selected_gpus}")
        else:
            print(f"{indent}Node (Prefix: {len(prefix_node.value)}) has no LP Node mapping")

        for child in prefix_node.children.values():
            self.pretty_print_helper(child, indent + "  ", depth=depth + 1)

    def update_nodes_with_solution(self):
        for prefix_node, lp_node in self.node_map.items():
            prefix_node.gpu_selections = set()
            for gpu_id, var in enumerate(lp_node.variables):
                if var.x >= 0.99:
                    prefix_node.gpu_selections.add(gpu_id)

    
class LPScheduler:
    def __init__(self, num_nodes: int, depth_limit=4, update_interval=5):
        self.num_nodes = num_nodes
        self.tree_cache = LPRadixCache()
        self.shadow_cache = LPRadixCache()
        self.lp_tree_traversal = LPTreeTraversal(num_nodes)
        self.lp_tree_traversal.depth_limit = depth_limit
        self.metrics_dict = []
        self.counter = 0
        self.update_interval=update_interval

    def runtime_selector(self, text: str=None, request_id: str=None, input_ids=None, ):
        # Tokenize the text
        start_time = time.time()

        node_map = self.lp_tree_traversal.node_map
        self.tree_cache.insert(tuple(input_ids), node_map=node_map)
        if self.counter < self.update_interval or self.counter % self.update_interval == 0:
            # Note this can be done in a background thread
            existing_cost = self.lp_tree_traversal.get_exisiting_cost()
            self.lp_tree_traversal.traverse_and_optimize(self.tree_cache.root_node, existing_cost=existing_cost)
            self.lp_tree_traversal.update_nodes_with_solution()

        self.counter += 1
        gpu_selections = self.tree_cache.match_prefix_get_gpu_selection(input_ids)

        # Randomly select a node from gpu selections
        mode = "not_random"
        if len(gpu_selections) == 0 or len(gpu_selections) == self.num_nodes:
            gpu_selections = set(range(self.num_nodes))
            mode = "random"

        runtime_selected = random.choice(list(gpu_selections))
        # Insert the tokenized text into the radix cache

        self.metrics_dict.append({
            "text": text,
            "rid": request_id,
            "selected_runtime": runtime_selected,
            "overhead": time.time() - start_time,
            "mode": mode
        })
        return runtime_selected


if __name__ == "__main__":
    # Example usage (you would need to define the structure of PrefixLPTreeNode and provide a valid prefix_tree_root):
    import sys
    import os
    import pandas as pd
    import copy

    # Add the parent directory of the 'src' directory to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from transformers import AutoTokenizer
    from benchmarks.benchmark_workload_gen import ToolBenchDataLoader, LoadDistribution
    num_workloads = 100
    num_requests = 4096
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataloader = ToolBenchDataLoader('benchmarks/G1_workload_updated_input_output_lengths_4096_cropped_to_50.json', num_workloads, num_requests, tokenizer, LoadDistribution.EVEN)
    workload = dataloader.generate_workload(k=1.1)

    print(f"Workload length: {len(workload)}")
    scheduler = LPScheduler(3, depth_limit=4)
    runtime_selected = []
    for i in range(5):
        runtime = scheduler.runtime_selector(input_ids=workload[i]["input_ids"], text=workload[i]["text"])
        runtime_selected.append(runtime)
    print(runtime_selected)

# %%


# %%


# %%



