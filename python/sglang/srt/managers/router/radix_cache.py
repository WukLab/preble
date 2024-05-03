import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List
import logging

import torch

logger = logging.getLogger(__name__)

@dataclass
class EvictionData():
    input_ids: list
    evicted_ids: list

class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.value = None
        self.ref_counter = 0
        self.last_access_time = time.time()

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class RadixCache:
    def __init__(self, disable=False, enable_partial_eviction=False):
        self.reset()
        self.disable = disable
        self.evicted_iteration = []
        self.enable_partial_eviction = enable_partial_eviction

    ##### Public API #####

    def flush_evicted(self):
        self.evicted_iteration = []

    def add_node_to_evicted_iteration(self, node, num_evited_token):
        input_ids = []
        evicted_ids = []
        while node != self.root_node:
            for k, v in node.parent.children.items():
                if v == node:
                    input_ids = list(k) + input_ids
                    if not evicted_ids:
                        evicted_ids = list(k[-num_evited_token:])
                    break
            node = node.parent
        self.evicted_iteration.append(EvictionData(input_ids, evicted_ids))
        

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.value = []
        self.root_node.ref_counter = 1
        self.evictable_size_ = 0
        self.evicted_iteration = []

    def match_prefix(self, key):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)
        if value:
            if isinstance(value[0], torch.Tensor):
                value = torch.concat(value)
            else:
                concatenated_value = []
                for v in value:
                    concatenated_value.extend(v)  # Assuming each element in value is a list itself
                value = concatenated_value
        return value, last_node[0]

    def match_prefix_return_str(self, key):
        return "".join(self.match_prefix(key)[0])

    def insert(self, key, value=None):
        if self.disable:
            return len(key)

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens, evict_callback, collect_evicted_node=False):
        # curr_evict = self.evictable_size()
        # start = time.perf_counter()
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
                
            holding_slots = len(x.value)
            desired_eviction = min(holding_slots, num_tokens - num_evicted) if self.enable_partial_eviction else holding_slots
            num_evicted += evict_callback(x.value[-desired_eviction:]).item()
            # logger.info(f'evicted: {desired_eviction} - {num_evicted}')
            self._delete_leaf(x, desired_eviction)
            
            if collect_evicted_node:
                self.add_node_to_evicted_iteration(x, desired_eviction)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)
        # end = time.perf_counter()
        # print(f"Eviction: {end-start}, Amount evicted: {curr_evict - self.evictable_size()}")
                
    def total_unique_kv_tokens(self, reqs):
        seen = set()
        
        # length of token is not in tree, calculated independently regardless
        def _traverse(node, length):
            while node != self.root_node:
                if node.ref_counter > 0:
                    seen.add(node)
                    length -= len(node.value)
                node = node.parent
            return length
            
        total_tokens = 0
        for req in reqs:
            total_tokens += _traverse(req.last_node, len(req.input_ids) + len(req.output_ids))
        for node in seen:
            total_tokens += len(node.value)
        return total_tokens
    
    def total_shared_token_by_count(self):
        def _helper(node: TreeNode):
            if node.ref_counter <= 1:
                return 0
            x = len(node.value) * (node.ref_counter - 1)
            for child in node.children.values():
                x += _helper(child)
            return x
        
        sum_shared = 0
        for child in self.root_node.children.values():
            sum_shared += _helper(child)    
        return sum_shared
    
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

    ##### Internal Helper Functions #####
    def _match_prefix_helper(self, node, key, value, last_node):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                if prefix_len < len(c_key):
                    new_node = self._split_node(c_key, child, prefix_len)
                    value.append(new_node.value)
                    last_node[0] = new_node
                else:
                    value.append(child.value)
                    last_node[0] = child
                    self._match_prefix_helper(child, key[prefix_len:], value, last_node)
                break

    def _split_node(self, key, child, split_len):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:]: child}
        new_node.parent = child.parent
        new_node.ref_counter = child.ref_counter
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len]] = new_node
        del new_node.parent.children[key]
        return new_node

    def _insert_helper(self, node, key, value):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)

            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            if prefix_len:
                new_node = self._split_node(c_key, child, prefix_len)
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:]
                )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.value = value
            node.children[key] = new_node
            self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node, indent):
        for key, child in node.children.items():
            print(" " * indent, len(key), key[:10], f"r={child.ref_counter}")
            self._print_helper(child, indent=indent + 2)

    #NOTE: tree node should not be deleted if partial eviction
    def _delete_leaf(self, node, num_evict_token):
        assert num_evict_token > 0, "num_evict_token should be greater than 0"
        for k, v in node.parent.children.items():
            if v == node:
                break
        
        del node.parent.children[k]
        if num_evict_token < len(node.value):
            node.value = node.value[:-num_evict_token]
            node.parent.children[k[:-num_evict_token]] = node
        self.evictable_size_ -= num_evict_token

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


if __name__ == "__main__":
    tree = RadixCache(disable=False)

    tree.insert("Hello")
    tree.insert("Hello There")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    print(tree.match_prefix_return_str("Hello T"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
