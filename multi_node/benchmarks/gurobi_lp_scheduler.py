# %%
import copy
import random

# Add the parent directory of the 'src' directory to the Python path


# %%
import heapq
import time
from collections import defaultdict
from uuid import uuid4  


class TreeNode:
    def __init__(self):
        self.id = uuid4()  
        self.children = defaultdict(TreeNode)
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

    def match_prefix_get_gpu_selection(self, key):
        if self.disable:
            return [], self.root_node

        value = []
        current_gpu_selection = self.root_node.gpu_selections
        current_gpu_selection, node = self._match_prefix_helper_gpu_selection(self.root_node, key, value, current_gpu_selection)
        return current_gpu_selection, node

    def _match_prefix_helper_gpu_selection(self, node, key, value, current_gpu_selection):
        node.last_access_time = time.time()
        child: TreeNode
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
        return current_gpu_selection, node

    def match_prefix_return_str(self, key):
        return "".join(self.match_prefix(key)[0])

    def insert(self, key, value=None, node_map=None, all_modified_nodes=None, split_nodes=None, depth_limit=0):
        if node_map is None:
            node_map = {}
            print("Node map is None")
        if split_nodes is None:
            split_nodes = {} # key -> node
        if self.disable:
            return len(key)

        if value is None:
            value = [x for x in key]
        modified_nodes = set()
        total_tokens_added = self._insert_helper(
            self.root_node,
            key, 
            value, 
            node_map=node_map, 
            modified_nodes=modified_nodes, 
            depth_limit=depth_limit, 
            current_depth=0,
            split_nodes=split_nodes
        )

        node: TreeNode
        for node in modified_nodes:
            # Add all parents till parent is none to all_modified_nodes
            while node is not None:
                all_modified_nodes.add(node)
                node = node.parent
        return total_tokens_added

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
        self.dec_ref_counter(node) # remove reference counter up to parent

    def evictable_size(self):
        return self.evictable_size_

    def _split_node(self, key, child, split_len, node_map, depth_limit, current_depth):
        # new_node -> child
        new_node = TreeNode()
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

        # if child in node_map and current_depth < depth_limit:
        #     node_map[new_node] = node_map[child]

        return new_node

    def _insert_helper(self, node, key, value, node_map, modified_nodes, depth_limit, current_depth, split_nodes):
        node.last_access_time = time.time()
        node.ref_counter += 1
        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)

            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    child.ref_counter += 1
                    modified_nodes.add(child)
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value, node_map=node_map, modified_nodes=modified_nodes, depth_limit=depth_limit, current_depth=current_depth + 1, split_nodes=split_nodes)

            if prefix_len:
                new_node = self._split_node(c_key, child, prefix_len, node_map, depth_limit=depth_limit, current_depth=current_depth + 1)
                modified_nodes.add(new_node)
                modified_nodes.add(child)
                if child in node_map and current_depth < depth_limit:
                    split_nodes[child] = new_node
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:], node_map=node_map, modified_nodes=modified_nodes, depth_limit=depth_limit, current_depth=current_depth + 1, split_nodes=split_nodes
                )

        if len(key):
            new_node = TreeNode()
            new_node.gpu_selections = copy.deepcopy(node.gpu_selections)
            new_node.parent = node
            new_node.value = value
            new_node.ref_counter = 1
            node.children[key] = new_node
            self.evictable_size_ += len(value)
            if current_depth < depth_limit:
                modified_nodes.add(new_node)
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

class LpNode:
    def __init__(self, node_id, num_gpus):
        self.node_id = node_id
        self.variables = [None for _ in range(num_gpus)]  # Will be initialized as binary variables in the model
        self.children_token_cost_at_max_depth = 0 # Issue is that depth_limit will cut off the tokens for children and that will treat it as free
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


# %%

from gurobipy import GRB
import gurobipy as gp
from typing import Optional
from collections import defaultdict

class LPGurobiTreeTraversal:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.node_map = {}
        self.depth_limit = 5
        # Create a new model. Gurobi doesn't use a solver_name argument; it automatically selects the best solver.
        self.model = gp.Model("LPTreeTraversal")
        self.model.setParam('OutputFlag', 0)  # Equivalent to verbose = 1 in python-mip
        self.model.setParam('LogToConsole', 0)
        self.total_cost_var = None
        self.per_gpu_constraints = []
        self.iteration_counter = 0 
        self.all_node_constraints = defaultdict(list)
        self.max_per_gpu_cost = self.model.addVar(name="max_per_gpu_cost", vtype=GRB.INTEGER)
        self.max_per_gpu_cost_constr = []
        self.existing_constraints = {}

    def init_load_vars(self, prefix_node: TreeNode, lp_node: LpNode, is_leaf: bool):
        if not is_leaf:
            return
        for gpu in range(self.num_gpus):
            lp_node.load_variables[gpu] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"l_{prefix_node.id}_{gpu}")
        lp_node.common_load = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"l_avg_{prefix_node.id}", lb=0)

    def init_lp_node(self, prefix_node, is_leaf):
        if prefix_node not in self.node_map:
            lp_node = LpNode(prefix_node.id, self.num_gpus)
            self.node_map[prefix_node] = lp_node
            for gpu in range(self.num_gpus):
                lp_node.variables[gpu] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{prefix_node.id}_{gpu}")
        lp_node = self.node_map[prefix_node]
        self.init_load_vars(prefix_node, lp_node, is_leaf)
        return lp_node

        # self.init_load_vars(current_lp_node, is_leaf)
        # # Initialize binary variables for the LP node

        # for gpu in range(self.num_gpus):
        #     current_lp_node.variables[gpu] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{prefix_node.id}_{gpu}")
        #     if is_leaf:
        #         current_lp_node.load_variables[gpu] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"l_{prefix_node.id}_{gpu}")
        # if is_leaf:
        #     current_lp_node.common_load = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"l_avg_{prefix_node.id}", lb=0)
        # return current_lp_node

    def _traverse_tree(self, current_prefix_node: TreeNode, parent_lp_node: Optional[LpNode]=None, depth=0, modified_nodes=None):
        if modified_nodes is not None and current_prefix_node not in modified_nodes:
            return  # Skip nodes that have not been modified

        if depth == self.depth_limit:
            assert parent_lp_node is not None
            parent_lp_node.children_token_cost_at_max_depth = self._calculate_children_token_cost(current_prefix_node)
            return
        
        is_leaf = not current_prefix_node.children or depth + 1 == self.depth_limit
        current_lp_node = self.init_lp_node(current_prefix_node, is_leaf)

        # At least one GPU must be allocated for a prefix, with a unique constraint name
        constraint_name = f"alloc_at_least_one_gpu_{current_prefix_node.id}"
        constr = self.model.addConstr(gp.quicksum(current_lp_node.variables) >= 1, name=constraint_name)
        self.all_node_constraints[current_prefix_node.id].append(constr)

        if is_leaf:        
            # Constrain the load on all the nodes
            for gpu_index in range(self.num_gpus):
                var = current_lp_node.variables[gpu_index]
                load_lower_constr = self.model.addConstr(current_lp_node.load_variables[gpu_index] >= var, name=f"load_var_{current_prefix_node.id}_{gpu_index}_lower")
                load_upper_constr = self.model.addConstr(current_lp_node.load_variables[gpu_index] <= var * current_prefix_node.ref_counter, name=f"load_var_{current_prefix_node.id}_{gpu_index}_upper")
                avg_load_upper_constr = self.model.addConstr(current_lp_node.load_variables[gpu_index] <= current_lp_node.common_load + 1000000 * (1 - var), name=f"load_var_{current_prefix_node.id}_{gpu_index}_common_load_upper")
                avg_load_lower_constr = self.model.addConstr(current_lp_node.load_variables[gpu_index] >= current_lp_node.common_load - 1000000 * (1 - var), name=f"load_var_{current_prefix_node.id}_{gpu_index}_common_load_lower")
                # self.all_node_constraints[current_prefix_node.id].extend([load_lower_constr, load_upper_constr, avg_load_upper_constr, avg_load_lower_constr])
                self.all_node_constraints[current_prefix_node.id].extend([load_upper_constr, avg_load_upper_constr, avg_load_lower_constr, load_lower_constr])
            
            constr = self.model.addConstr(gp.quicksum(current_lp_node.load_variables) >= current_prefix_node.ref_counter, name=f"load_constr_sum_{current_prefix_node.id}")
            self.all_node_constraints[current_prefix_node.id].append(constr)

        if parent_lp_node:
            # If the child takes a node, then the parent must also take a node, with unique constraint names for each GPU
            for gpu in range(self.num_gpus):
                constraint_name = f"parent_child_gpu_alloc_{current_lp_node.node_id}_{parent_lp_node.node_id}_{gpu}"
                constr = self.model.addConstr(current_lp_node.variables[gpu] <= parent_lp_node.variables[gpu], name=constraint_name)
                self.all_node_constraints[current_prefix_node.id].append(constr)
        for child_prefix_node in current_prefix_node.children.values():
            self._traverse_tree(current_prefix_node=child_prefix_node, parent_lp_node=current_lp_node, depth=depth + 1, modified_nodes=modified_nodes)


    def add_parent_child_gpu_constraints(self, modified_nodes=None):
        for parent_prefix_node, parent_lp_node in self.node_map.items():
            if modified_nodes is not None and parent_prefix_node not in modified_nodes:
                continue
            for gpu_index in range(self.num_gpus):
                children_gpu_selections = []
                for child_prefix_node in parent_prefix_node.children.values():
                    if child_prefix_node in self.node_map:
                        child_lp_node = self.node_map[child_prefix_node]
                        children_gpu_selections.append(child_lp_node.variables[gpu_index])
                if children_gpu_selections:
                    # Use Gurobi's quicksum for summing up the selections
                    children_selections_total = gp.quicksum(children_gpu_selections)
                    # Create a constraint with a unique name for easy identification
                    constraint_name = f"parent_{parent_lp_node.node_id}_gpu_{gpu_index}_children_selection"
                    constr = self.model.addConstr(parent_lp_node.variables[gpu_index] <= children_selections_total, name=constraint_name)
                    self.all_node_constraints[parent_prefix_node.id].append(constr)

    # def add_parent_child_gpu_constraints_recursively(self, current_node, gpu_assignment_variables=None, depth=0):
    #     if gpu_assignment_variables is None:
    #         gpu_assignment_variables = {gpu_index: [] for gpu_index in range(self.num_gpus)}
        
    #     if depth == self.depth_limit:
    #         return gpu_assignment_variables

    #     if current_node not in modified_nodes:
    #         return gpu_assignment_variables

    #     # Base case: if the current node has no children, return the GPU assignment variables for the current node
    #     if not current_node.children:
    #         for gpu_index in range(self.num_gpus):
    #             gpu_assignment_variables[gpu_index].append(self.node_map[current_node].variables[gpu_index])
    #         return gpu_assignment_variables

    #     # Recursive case: accumulate GPU assignment variables from children
    #     for child in current_node.children.values():
    #         child_gpu_assignments = self.add_parent_child_gpu_constraints_recursively(child, gpu_assignment_variables, depth=depth+1)

    #     # After collecting assignments from children, apply constraints for the current node
    #     for gpu_index in range(self.num_gpus):
    #         if current_node in self.node_map:  # Check if current node is in the LP model
    #             current_lp_node = self.node_map[current_node]
    #             # Sum children GPU assignments for the current GPU
    #             children_selections_total = gp.quicksum(child_gpu_assignments[gpu_index])
    #             # Add constraint for the current node on the current GPU
    #             constraint_name = f"parent_{current_lp_node.node_id}_gpu_{gpu_index}_children_selection"
    #             constr = self.model.addConstr(current_lp_node.variables[gpu_index] <= children_selections_total, name=constraint_name)
    #             self.all_node_constraints[current_node.id].append(constr)
    #     return gpu_assignment_variables

    def traverse_and_optimize(self, prefix_tree_root, previous_cost={}, modified_nodes=None, objective_only=False):
        start_time = time.time()
        
        for node in modified_nodes:
            for constr in self.all_node_constraints[node.id]:
                self.model.remove(constr)
            self.all_node_constraints[node.id] = []
        
        for constr in self.max_per_gpu_cost_constr:
            self.model.remove(constr)
        
        self.max_per_gpu_cost_constr = []

        self._traverse_tree(prefix_tree_root, modified_nodes=modified_nodes)  # Set up variables and base constraints
        constraint_removal_time2 = time.time() - start_time
        
        start_time = time.time()
        self.add_parent_child_gpu_constraints(modified_nodes=modified_nodes)  # Add parent-child constraints
        # self.add_parent_child_gpu_constraints_recursively(prefix_tree_root)
        constraint_removal_time = time.time() - start_time

        start_time = time.time()

        # Objective components: Let's assume we're trying to minimize the total cost adjusted for existing costs
        total_cost = gp.LinExpr()
        per_gpu_load_cost = [gp.LinExpr() for _ in range(self.num_gpus)]
        per_gpu_mem_load_cost = [gp.LinExpr() for _ in range(self.num_gpus)]

        initial_solution = []
        for prefix_node, lp_node in self.node_map.items():
            num_tokens_total = prefix_node.num_tokens + lp_node.children_token_cost_at_max_depth
            for gpu_index, var in enumerate(lp_node.variables):
                previous_gpu_selected = previous_cost['placement_cost'].get(prefix_node, {}).get(gpu_index, 0)
                if previous_gpu_selected:
                    var.Start = previous_gpu_selected  # Set initial solution
                mistral_tokens_to_prefill_time = lambda x: 0.148 * x + 22.7
                num_tokens_time = mistral_tokens_to_prefill_time(num_tokens_total)
                
                decode_length = 16  # Assume decoding occurs for 20 tokens
                decoding_time = lambda x: 6.7 * x
                total_cost += var * (num_tokens_time - previous_gpu_selected * num_tokens_time)

                if lp_node.load_variables[gpu_index]:
                    load_cost = lp_node.load_variables[gpu_index] * decoding_time(decode_length) 
                    per_gpu_load_cost[gpu_index] += load_cost
                # per_gpu_load_cost[gpu_index] += var * decoding_time(decode_length)
                if prefix_node.ref_counter > 1: #Avoid double counting
                    per_gpu_mem_load_cost[gpu_index] += var * num_tokens_time

        for gpu_id, item in enumerate(per_gpu_load_cost):
            constr = self.model.addConstr(item + per_gpu_mem_load_cost[gpu_index] <= self.max_per_gpu_cost, name=f"max_per_gpu_cost_constr_{gpu_id}")
            self.max_per_gpu_cost_constr.append(constr)

        # Set objective
        setup_time = time.time() - start_time
        self.model.setObjective(total_cost + self.max_per_gpu_cost, GRB.MINIMIZE)

        # Model parameters
        self.model.setParam('Threads', 0)
        self.model.setParam('TimeLimit', 0.05)
        self.model.setParam('MIPGap', 0.02)

        solving_start_time = time.time()
        self.model.optimize()
        solving_time = time.time() - solving_start_time


        if self.model.Status == GRB.OPTIMAL:
            # print('Optimal solution found.')
            pass
        elif self.model.Status == GRB.INFEASIBLE:
            print('Infeasable solution found')

        else:
            print('Feasible solution found.')

        # tokens_per_gpu, load_to_gpu = self.calculate_tokens_per_gpu()
        # print(f"Tokens per GPU: {tokens_per_gpu}, Load to GPU: {load_to_gpu}")
        # print(f"Objective value: {self.model.ObjVal}")
        # 
        # To get the total number of variables in the model
        num_vars = self.model.numVars

        # To get the total number of constraints in the model
        num_constraints = self.model.numConstrs

        # Print total number of parameters (variables and constraints)
        # print(f"Total number of variables: {num_vars}")
        # print(f"Total number of constraints: {num_constraints}")

        print(f"Solving time: {solving_time}s, Setup Time: {setup_time}s Total constraint_removal_time {constraint_removal_time}s, {constraint_removal_time2} modified nodes len {len(modified_nodes)}, {num_vars}, {num_constraints}")
        return time.time() - start_time

    def get_previous_cost(self, split_nodes={}):
        placement_cost = {}
        load_cost = {}
        for prefix_node, lp_node in self.node_map.items():
            load_cost[prefix_node] = {}
            placement_cost[prefix_node] = {}
            for gpu_id, var in enumerate(lp_node.variables):
                # Initialize a nested dictionary if the prefix node hasn't been encountered yet
                # Check if the variable was selected by the solver or randomly, for Gurobi use var.X to get the solution value
                solver_selection = var and var.X >= 0.99  # Gurobi uses var.X to access the value of a variable
                random_selection = lp_node.randomly_selected_gpu is not None and lp_node.randomly_selected_gpu == gpu_id
                if solver_selection or random_selection:
                    placement_cost[prefix_node][gpu_id] = 1
                else:
                    placement_cost[prefix_node][gpu_id] = 0
                if lp_node.load_variables[gpu_id]:
                    load_cost[prefix_node][gpu_id] = lp_node.load_variables[gpu_id].X
        # Handle split nodes by preserving the existing cost across the split
        if split_nodes:
            for node_id, new_node_id in split_nodes.items():
                if node_id in placement_cost:
                    placement_cost[new_node_id] = placement_cost[node_id]
        return {
            "placement_cost": placement_cost,
            "load_cost": load_cost
        }

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

    def calculate_tokens_per_gpu(self):
        tokens_per_gpu = {gpu: 0 for gpu in range(self.num_gpus)}  # Reset/initialize
        load_to_gpu = {gpu: 0 for gpu in range(self.num_gpus)}
        lp_node:LpNode
        # pretty print node map
        for prefix_node, lp_node in self.node_map.items():
            for i, var in enumerate(lp_node.variables):
                solved_var = var.X if var.X >= 0.99 else 0
                if solved_var:  # If GPU i is selected by this node, using .x for variable value in MIP
                    tokens_per_gpu[i] += prefix_node.num_tokens  # Accumulate tokens
                    load_to_gpu[i] += prefix_node.ref_counter
        return tokens_per_gpu, load_to_gpu

    def pretty_print(self, prefix_node):
        self.pretty_print_helper(prefix_node)

    def pretty_print(self, prefix_node):
        self.pretty_print_helper(prefix_node)

    def pretty_print_helper(self, prefix_node, indent="", depth=0):
        if depth == self.depth_limit:
            return
        lp_node = self.node_map.get(prefix_node)
        if lp_node:
            if lp_node.randomly_selected_gpu:
                selected_gpus = [lp_node.randomly_selected_gpu]
            else:
                selected_gpus = [i for i, var in enumerate(lp_node.variables) if var and var.X >= 0.99]  # Adjust threshold as needed, using .x for variable value
            load_vars = [var.X if var else None for var in lp_node.load_variables]
            common_load = lp_node.common_load.X if lp_node.common_load else 0
            def get_tool(workload_item):
                text = tokenizer.decode(workload_item)
                if ":" in text:
                    return text.split(":")[0].strip().replace("\n", " ")
                else:
                    return text[:60].strip().replace("\n", "")
            print(f"{indent}Node {lp_node.node_id} (Tokens: {get_tool(prefix_node.value)}, {len(prefix_node.value)}): GPUs {selected_gpus}, Load {load_vars} Common Load {common_load}")
        else:
            print(f"{indent}Node (Prefix: {len(prefix_node.value)}) has no LP Node mapping")

        for child in prefix_node.children.values():
            self.pretty_print_helper(child, indent + "  ", depth=depth + 1)


    def update_nodes_with_solution(self, modified_nodes=None):
        for prefix_node, lp_node in self.node_map.items():
            prefix_node.gpu_selections = set()
            for gpu_id, var in enumerate(lp_node.variables):
                if var.X>= 0.99:
                    prefix_node.gpu_selections.add(gpu_id)

class GurobiLPScheduler:
    def __init__(self, num_nodes: int, depth_limit=4):
        self.num_nodes = num_nodes
        self.tree_cache = RadixCache()
        self.lp_tree_traversal = LPGurobiTreeTraversal(num_nodes)
        self.lp_tree_traversal.depth_limit = depth_limit
        self.metrics_dict = []
        self.counter = 0
        self.load = {

        }
        self.modified_nodes = set()

    def runtime_selector(self, text: str=None, request_id: str=None, input_ids=None, ):
        # Tokenize the text
        start_time = time.time()

        node_map = self.lp_tree_traversal.node_map
        split_nodes = {}
        self.tree_cache.insert(tuple(input_ids), node_map=node_map, all_modified_nodes=self.modified_nodes, depth_limit=self.lp_tree_traversal.depth_limit, split_nodes=split_nodes)
        existing_cost = self.lp_tree_traversal.get_exisiting_cost(split_nodes)
        self.lp_tree_traversal.traverse_and_optimize(self.tree_cache.root_node, existing_cost=existing_cost, modified_nodes=self.modified_nodes)
        self.lp_tree_traversal.update_nodes_with_solution()
        self.modified_nodes = set()

        self.counter += 1
        gpu_selections, node = self.tree_cache.match_prefix_get_gpu_selection(input_ids)
        # Randomly select a node from gpu selections
        mode = "not_random"
        if len(gpu_selections) == 0 or len(gpu_selections) == self.num_nodes:
            print("Random selection", gpu_selections)
            gpu_selections = set(range(self.num_nodes))
            mode = "random"

        runtime_selected = random.choice(list(gpu_selections))
        self.load[runtime_selected] = self.load.get(runtime_selected, 0) + 1
        self.metrics_dict.append({
            "text": text,
            "rid": request_id,
            "selected_runtime": runtime_selected,
            "overhead": time.time() - start_time,
            "mode": mode
        })
        return runtime_selected
