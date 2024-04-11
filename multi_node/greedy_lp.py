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


class TreeNode:
    def __init__(self):
        self.id = uuid4()
        self.children = defaultdict(TreeNode)
        self.parent: Optional[TreeNode] = None
        self.value = None
        self.ref_counter = 0
        self.last_access_time = time.time()
        self.gpu_selections = set()
        self.is_leaf = False

    @property
    def num_tokens(self):
        return len(self.value)

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return self.id == other.id  # Compare nodes based on their unique ID
        return False

    def __hash__(self):
        return hash(self.id)  # Use the unique ID for hashing

    def __repr__(self) -> str:
        return f"TreeNode(id={self.id}, ref_counter={self.ref_counter})"


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class RadixCache:
    def __init__(self, disable=False):
        self.reset()
        self.disable = disable

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
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
        node.last_access_time = time.time()
        child: TreeNode
        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                if child.gpu_selections:
                    current_gpu_selection = child.gpu_selections
                if prefix_len < len(c_key):
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
            print("Node map is None")
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

        node: TreeNode = created_node
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

    def remove_completed_input_ids(self, input_ids):
        node = self.find_node(input_ids)
        self.dec_ref_counter(node)  # remove reference counter up to parent

    def evictable_size(self):
        return self.evictable_size_

    def _split_node(
        self, key, child: TreeNode, split_len, node_map, depth_limit, current_depth
    ):
        # new_node -> child
        new_node = TreeNode()
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
        node: TreeNode,
        key,
        value,
        node_map,
        modified_nodes,
        depth_limit,
        current_depth,
        split_nodes,
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
                if child in node_map and current_depth < depth_limit:
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
                )

        if len(key):
            new_node = TreeNode()
            new_node.gpu_selections = copy.deepcopy(node.gpu_selections)
            new_node.parent = node
            new_node.value = value
            new_node.ref_counter = 1
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
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.node_map = defaultdict(set)
        self.depth_limit = 3
        self.current_load_cost = [0 for _ in range(num_gpus)]
        self.current_memory_cost = [0 for _ in range(num_gpus)]

    def _calculate_children_token_cost(self, node: TreeNode):
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

    def traverse_and_optimize(
        self, leaf_node: TreeNode, modified_nodes: set[TreeNode] = None, split_nodes={}
    ):
        start_time = time.time()

        self.model = gp.Model("LPTreeTraversal")
        self.model.setParam("OutputFlag", 0)  # Equivalent to verbose = 1 in python-mip
        self.model.setParam("LogToConsole", 0)

        for key, value in split_nodes.items():
            self.node_map[value] = self.node_map[key]

        self.max_per_gpu_cost_constr = []

        lp_node = LpNode("main", self.num_gpus)
        for gpu in range(self.num_gpus):
            lp_node.variables[gpu] = self.model.addVar(
                vtype=GRB.BINARY, name=f"x_{gpu}"
            )

        self.model.update()

        self.model.addConstr(
            gp.quicksum(lp_node.variables) >= 1
        )  # at least 1 variable should be one
        # Objective components: Let's assume we're trying to minimize the total cost adjusted for existing costs
        total_cost = gp.LinExpr()
        per_gpu_load_cost = [gp.LinExpr() for _ in range(self.num_gpus)]
        per_gpu_mem_load_cost = [gp.LinExpr() for _ in range(self.num_gpus)]

        new_total_memory_cost = [0 for _ in range(self.num_gpus)]

        decode_length = 45  # Assume decoding occurs for 20 tokens
        decoding_time = lambda x: 6.7 * x
        total_decode_time = decoding_time(decode_length)

        for prefix_node in modified_nodes:
            num_tokens_total = 0
            if prefix_node == leaf_node:
                num_tokens_total += self._calculate_children_token_cost(leaf_node)
            else:
                num_tokens_total += prefix_node.num_tokens

            mistral_tokens_to_prefill_time = lambda x: 0.148 * x + 22.7
            num_tokens_time = mistral_tokens_to_prefill_time(num_tokens_total)

            for gpu_index, var in enumerate(lp_node.variables):
                previous_gpu_selected = gpu_index in self.node_map[prefix_node]
                recomp_cost = var * (
                    num_tokens_time - previous_gpu_selected * num_tokens_time
                )
                total_cost += recomp_cost
                new_total_memory_cost[gpu_index] += (
                    num_tokens_time - previous_gpu_selected * num_tokens_time
                )

        for gpu_index in range(self.num_gpus):
            # Increment load by decoding time each time
            per_gpu_load_cost[gpu_index] += (
                lp_node.variables[gpu_index] * total_decode_time
            )

        max_per_gpu_cost = self.model.addVar(name="max_per_gpu_cost", vtype=GRB.INTEGER)
        for gpu_index in range(self.num_gpus):
            per_gpu_mem_load_cost[gpu_index] += self.current_memory_cost[gpu_index]
            # per_gpu_load_cost[gpu_index] += self.current_load_cost[gpu_index]
            self.model.addConstr(
                per_gpu_mem_load_cost[gpu_index] + per_gpu_load_cost[gpu_index]
                <= max_per_gpu_cost,
                name=f"max_per_gpu_cost_constr_{gpu_index}",
            )
            # updated load cost
        # Set objective
        self.model.setObjective(max_per_gpu_cost + total_cost, GRB.MINIMIZE)

        # Model parameters
        self.model.setParam("Threads", 0)
        self.model.setParam("TimeLimit", 0.05)
        self.model.setParam("MIPGap", 0.02)

        self.model.optimize()
        self.model.update()
        if self.model.Status == GRB.OPTIMAL:
            # print('Optimal solution found.')
            pass
        elif self.model.Status == GRB.INFEASIBLE:
            print("Infeasable solution found")

        else:
            pass
            # print('Feasible solution found.')

        # todo find placement
        selected_gpus = [
            gpu_id for gpu_id, var in enumerate(lp_node.variables) if var.X >= 0.99
        ]
        leaf_node.gpu_selections = set(selected_gpus)
        self.node_map[leaf_node] = leaf_node.gpu_selections

        node: TreeNode = leaf_node.parent
        while node != None:
            parent_gpu_selection = set()
            for key, children in node.children.items():
                parent_gpu_selection.update(children.gpu_selections)
            node.gpu_selections = parent_gpu_selection
            self.node_map[node] = parent_gpu_selection
            node = node.parent

        for gpu in selected_gpus:
            self.current_load_cost[
                gpu
            ] += total_decode_time  # Increase by decoding time to each gpu
            self.current_memory_cost[gpu] += new_total_memory_cost[gpu]

        # To get the total number of variables in the model
        num_vars = self.model.numVars

        # To get the total number of constraints in the model
        num_constraints = self.model.numConstrs

        # Print total number of parameters (variables and constraints)
        # print(f"Total number of variables: {num_vars}")
        # print(f"Total number of constraints: {num_constraints}")

        # print(f"Solving time: {(time.time() - start_time) * 1000}ms")
        return time.time() - start_time

    def pretty_print(self, prefix_node, depth_limit=4, tokenizer=None):
        self.pretty_print_helper(
            prefix_node, depth_limit=depth_limit, tokenizer=tokenizer
        )

    def pretty_print_helper(
        self, prefix_node: TreeNode, indent="", depth=0, depth_limit=4, tokenizer=None
    ):
        if depth == depth_limit:
            return
        selected_gpus = self.node_map.get(prefix_node)

        def get_tool(workload_item):
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
        for prefix_node, lp_node in self.node_map.items():
            prefix_node.gpu_selections = set()
            for gpu_id, var in enumerate(lp_node.variables):
                if var.X >= 0.99:
                    prefix_node.gpu_selections.add(gpu_id)

    def completed_request(self, tree_cache, input_ids):
        decode_length = 45  # Assume decoding occurs for 20 tokens
        decoding_time = lambda x: 6.7 * x
        total_decode_time = decoding_time(decode_length)
        node: TreeNode = tree_cache.find_node(input_ids)
        tree_cache.remove_completed_input_ids(input_ids)
        for selection in node.gpu_selections:
            self.current_load_cost[selection] -= total_decode_time


class GurobiGreedyLPScheduler:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.tree_cache = RadixCache()
        self.lp_tree_traversal = LPGurobiGreedyTraversal(num_nodes)
        self.lp_tree_traversal.depth_limit = 64
        self.metrics_dict = []
        self.counter = 0
        self.load = {}
        self.lock = threading.Lock()
        self.modified_nodes = set()

    def runtime_selector(
        self,
        text: str = None,
        request_id: str = None,
        input_ids=None,
    ):
        # Tokenize the text
        start_time = time.time()
        with self.lock:
            node_map = self.lp_tree_traversal.node_map
            split_nodes = {}
            node = self.tree_cache.insert(
                tuple(input_ids),
                node_map=node_map,
                all_modified_nodes=self.modified_nodes,
                depth_limit=self.lp_tree_traversal.depth_limit,
                split_nodes=split_nodes,
            )
            self.lp_tree_traversal.traverse_and_optimize(
                node, modified_nodes=self.modified_nodes, split_nodes=split_nodes
            )
            gpu_selections = node.gpu_selections
            self.modified_nodes = set()

        self.counter += 1
        # Randomly select a node from gpu selections
        mode = "not_random"
        if len(gpu_selections) == 0 or len(gpu_selections) == self.num_nodes:
            gpu_selections = set(range(self.num_nodes))
            mode = "random"

        runtime_selected = random.choice(list(gpu_selections))
        self.load[runtime_selected] = self.load.get(runtime_selected, 0) + 1
        self.metrics_dict.append(
            {
                "text": text,
                "rid": request_id,
                "selected_runtime": runtime_selected,
                "overhead": time.time() - start_time,
                "mode": mode,
            }
        )
        return runtime_selected

    def finish_request(
        self, text: str = None, request_id: str = None, input_ids=None, func_output=None
    ):
        with self.lock:
            self.lp_tree_traversal.completed_request(self.tree_cache, input_ids)


if __name__ == "__main__":
    import random
    from transformers import AutoTokenizer
    import sys
    import os
    import copy
    import random

    # Add the parent directory of the 'src' directory to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname("."), "..")))
    from transformers import AutoTokenizer
    from benchmarks.benchmark_workload_gen import ToolBenchDataLoader, LoadDistribution

    cache = RadixCache()

    num_workloads = 100
    num_requests = 4096
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    random.seed(5)
    dataloader = ToolBenchDataLoader(
        "benchmarks/datasets/G1_workload_updated_input_output_lengths_4096_cropped_to_50.json",
        num_workloads,
        num_requests,
        tokenizer,
        LoadDistribution.ZIPF,
    )
    workload = dataloader.generate_workload(k=1.1)

    scheduler = GurobiGreedyLPScheduler(2)
    for i, item in enumerate(workload[:64]):
        runtime_selected = scheduler.runtime_selector(
            text=item["text"], request_id=i, input_ids=item["input_ids"]
        )
        # print(item["text"], runtime_selected)
    # print(pd.DataFrame(scheduler.metrics_dict))
    scheduler.lp_tree_traversal.pretty_print(
        scheduler.tree_cache.root_node, depth_limit=3, tokenizer=tokenizer
    )
    # breakpoint()
    # scheduler.lp_tree_traversal.pretty_print(scheduler.tree_cache.root_node, depth_limit=3)
