import sys
import os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpu_stats_profiling import Monitor

import unittest
import time
import cupy as cp
from data_parallel_request_cache import DataParallelRequestRouter, DataParallelRuntimeSelectionPolicy
from parameterized import parameterized
from sglang_experiments.sglang.multi_node.benchmarks.benchmark_workload_gen import get_react_workload, generate_random_workload
import random
from multi_node_loader import MultiNodeLoader
import sglang as sgl

# Test basic workload
#     tests = [
#     "Workload 1. Test 1",
#     "Workload 1. Test 2",
#     "Workload 1. Test 3",
#     "Workload 2. Test 1",
#     "Workload 2. Test 2",
#     "Workload 2. Test 3",
# ]

# Test 2 Long ReAct workloads get evenly Distributed between 2 Nodes

# Test N Long workloads get evenly distributed between 2 + Nodes
class TestDataParalelRouter(unittest.TestCase):

    @parameterized.expand([
        [DataParallelRuntimeSelectionPolicy.RANDOM],
        [DataParallelRuntimeSelectionPolicy.CONSISTENT_HASH]
    ])
    def test_valid_interface(self, policy):
        router = DataParallelRequestRouter(policy, 5)
        assert 0 <= router.select_runtime("test", 0, 0) < 5

# Workload 1: 



# TODO test timeit decorator
if __name__ == "__main__":
    unittest.main()
