import unittest
import os
os.environ['GRB_LICENSE_FILE'] = '/home/vikranth/gurobi.lic'
os.environ['GRB_LOGFILE'] = '/dev/null' 

from global_policy_lp import LPTreeTraversal, RadixCache
from benchmarks.benchmark_workload_gen import ToolBenchDataLoader, LoadDistribution, RandomDataLoader
from transformers import AutoTokenizer
import concurrent
from parameterized import parameterized


# Test the LPTreeTraversal with some sample inputs and verify the outputs
class TestLPTreeTraversal(unittest.TestCase):
    def setUp(self) -> None:
        num_workloads = 100
        num_requests = 4096
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        # dataloader = ToolBenchDataLoader('benchmarks/datasets/G1_workload_updated.json', num_workloads, num_requests, self.tokenizer, LoadDistribution.EVEN)
        return super().setUp()

    def tokenize_in_parallel(self, texts):
        # Use ThreadPool executor to tokenize the text
        with concurrent.futures.ThreadPoolExecutor(128) as executor:
            input_ids = list(executor.map(self.tokenizer.encode, texts))
        return input_ids

    @parameterized.expand([
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
    ])
    def test_lptree_traversal_basic_N_node(self, num_gpus):
        self._test_lptree_traversal_basic_N_node(num_gpus)

    def _test_lptree_traversal_basic_N_node(self, num_gpus, verbose=False):
        """
        If there are N strings of the form GPU: X with no difference in load. 
        They should uniquely be routed to individual gpus.

        An unknown prefix should get routed randomly to one of the gpus randomly(gpu selections all)
        """
        lp_tree_traversal = LPTreeTraversal(num_gpus)
        texts = [
            f"GPU {i} This is a sample sentence on gpu {i}" for i in range(num_gpus)
        ]
        input_ids = self.tokenize_in_parallel(texts)
        cache = RadixCache()
        for i in range(num_gpus):
            cache.insert(tuple(input_ids[i]))
        lp_tree_traversal.traverse_and_optimize(cache.root_node)
        lp_tree_traversal.update_nodes_with_solution()
        if verbose:
            lp_tree_traversal.pretty_print(cache.root_node)
        all_gpu_selections = set()
        for i in range(num_gpus):
            gpu_selections = cache.match_prefix_get_gpu_selection(input_ids[i])
            self.assertEqual(len(gpu_selections), 1)
            selected_gpu = gpu_selections.pop()
            self.assertFalse(selected_gpu in all_gpu_selections)
            all_gpu_selections.add(selected_gpu)
        # Test that if none of prefixes exist an empty set is returned
        text = "Unknown Prefix and text"
        unknown_input_id = self.tokenize_in_parallel([text])[0]
        node_map = lp_tree_traversal.node_map
        cache.insert(tuple(unknown_input_id), node_map=node_map)
        gpu_selection = cache.match_prefix_get_gpu_selection(unknown_input_id)
        self.assertEqual(gpu_selection, set(i for i in range(num_gpus)))

    def _test_basic_workload_branching_consistent_across_depths(self, depth_limit=3, verbose=False):
        lp_tree_traversal = LPTreeTraversal(2)
        lp_tree_traversal.depth_limit = depth_limit
        texts = [
            "Workload 1. A B C D",
            "Workload 2. A B C D", 
            "Workload 1. A B C D example 1",
            "Workload 1. A B C D example 2",
            "Workload 1. A B C D example 2 example 3",
            "Workload 2. A B C D E",
            "Workload 2. A B C D E F",
            "Workload 2. A B C D E F G",
        ]
        input_ids = [self.tokenizer.encode(text) for text in texts]
        cache = RadixCache()
        runtime_selected_per_workload = {
            "Workload 1": [],
            "Workload 2": []
        }
        for i in range(len(texts)):
            node_map = lp_tree_traversal.node_map
            cache.insert(tuple(input_ids[i]), node_map=node_map)
            existing_cost = lp_tree_traversal.get_exisiting_cost()
            lp_tree_traversal.traverse_and_optimize(cache.root_node, existing_cost=existing_cost)
            lp_tree_traversal.update_nodes_with_solution()
            # Placement to sentence 1 after the first iteration should be constant to the same gpu
            gpu_selections = cache.match_prefix_get_gpu_selection(input_ids[i])
            self.assertEqual(len(gpu_selections), 1)
            if "Workload 1" in texts[i]:
                runtime_selected_per_workload["Workload 1"].append(gpu_selections.pop())
            else:
                runtime_selected_per_workload["Workload 2"].append(gpu_selections.pop())
            if verbose:
                lp_tree_traversal.pretty_print(cache.root_node, tokenizer=self.tokenizer)
        # Test that the same workloads get the same runtime
        self.assertEqual(len(set(runtime_selected_per_workload["Workload 1"])), 1)
        self.assertEqual(len(set(runtime_selected_per_workload["Workload 2"])), 1)
        # Check that the workloads are placed on different runtimes
        self.assertNotEqual(runtime_selected_per_workload["Workload 1"][0], runtime_selected_per_workload["Workload 2"][0])

    @parameterized.expand([
        (3,),
        (4,),
        (5,),
        (6,),
    ])
    def test_basic_workload_branching_consistent_across_depths(self, depth_limit):
        self._test_basic_workload_branching_consistent_across_depths(depth_limit)

def run_tests():
    with open('test_output.log', 'a+') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)
if __name__ == "__main__":
    # test = TestLPTreeTraversal()
    # test.setUp()
    # test._test_lptree_traversal_basic_N_node(5)
    run_tests()