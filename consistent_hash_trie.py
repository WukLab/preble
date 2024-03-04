import hashlib
from sglang.srt.managers.router.radix_cache import RadixCache
import bisect
import random
from uhashring import HashRing

class SimpleHash:
    def __init__(self, num_nodes=2):
        self.num_nodes = num_nodes
        self.hash_ring = HashRing(nodes=list(range(num_nodes)))
        self.node_allocations = {node: [] for node in range(num_nodes)}
    def _hash(self, key):
        return hash(key)

    def get_node(self, key):
        node = self.hash_ring.get_node(key)
        self.node_allocations[node].append(key)
        return node

class ConsistentHashingWithRadixCache:
    def __init__(self, num_nodes=2):
        self.radix_cache = RadixCache()  # The provided RadixCache instance
        self.consistent_hashing = SimpleHash(num_nodes)

    def insert_key(self, key, value=None):
        longest_matching_prefix = self.radix_cache.match_prefix_return_str(key)
        self.radix_cache.insert(key, value)
        if not longest_matching_prefix:
            longest_matching_prefix = key
        # Map the trie node to a consistent hashing node
        node_id = self.consistent_hashing.get_node(longest_matching_prefix)
        return node_id

    def get_node_for_key(self, key):
        longest_matching_prefix = self.radix_cache.match_prefix_return_str(key)
        return self.consistent_hashing.get_node(longest_matching_prefix)

    def add_node(self, node_id):
        # Dynamically add a node to the consistent hashing ring
        self.consistent_hashing.add_node(node_id)
        print(f"Added node: {node_id}")

    def remove_node(self, node_id):
        # Dynamically remove a node from the consistent hashing ring
        self.consistent_hashing.remove_node(node_id)
        print(f"Removed node: {node_id}")

    def print_status(self):
        # Print the current status of the Radix Trie and the consistent hashing mapping
        print("Current RadixCache Status:")
        self.radix_cache.pretty_print()
        # Here you would also print the status of the ConsistentHashing instance, such as node distribution


if __name__ == "__main__":
    trie = ConsistentHashingWithRadixCache(num_nodes=2)
    tests = [
        "Workload 1. Test 1",
        "Workload 1. Test 2",
        "Workload 1. Test 3",
        "Workload 2. Test 1",
        "Workload 2. Test 2",
        "Workload 2. Test 3",
    ]
    random.shuffle(tests)
    for key in tests:
        print(key, trie.insert_key(key))
