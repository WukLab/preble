import unittest
import os
os.environ['GRB_LICENSE_FILE'] = '/home/vikranth/gurobi.lic'
os.environ['GRB_LOGFILE'] = '/dev/null' 

from greedy_lp import LPGurobiGreedyTraversal, LPRadixCache
# from greedy_lp_old import LPGurobiGreedyTraversal
# from greedy_lp_old import RadixCache as LPRadixCache
from benchmarks.benchmark_workload_gen import ToolBenchDataLoader, LoadDistribution, WorkloadPrefixDataLoader
from transformers import AutoTokenizer
import concurrent
from parameterized import parameterized
from sglang.srt.managers.router.model_runner import GPUConfig
from benchmarks.exp_configs.react_simulator_config import add_simulation_to_gpu_config


# Test the LPTreeTraversal with some sample inputs and verify the outputs
class TestLPTreeTraversal(unittest.TestCase):
    def setUp(self) -> None:
        num_workloads = 100
        num_requests = 4096
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.gpu_config = [
            GPUConfig(gpu_id=0, url=None, use_ssh=False),
            GPUConfig(gpu_id=0, url=None, use_ssh=False),
        ]
        add_simulation_to_gpu_config(self.gpu_config)
        self.lorem = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id
est laborum."""
    
        # dataloader = ToolBenchDataLoader('benchmarks/datasets/G1_workload_updated.json', num_workloads, num_requests, self.tokenizer, LoadDistribution.EVEN)
        return super().setUp()

    def tokenize_in_parallel(self, texts):
        # Use ThreadPool executor to tokenize the text
        with concurrent.futures.ThreadPoolExecutor(128) as executor:
            input_ids = list(executor.map(self.tokenizer.encode, texts))
        return input_ids

    # @parameterized.expand([
    #     (1,),
    #     (2,),
    #     (3,),
    #     (4,),
    #     (5,),
    # ])
    # def test_lptree_traversal_basic_N_node(self, num_gpus):
    #     self._test_lptree_traversal_basic_N_node(num_gpus)

    # def _test_lptree_traversal_basic_N_node(self, num_gpus, verbose=False):
    #     """
    #     If there are N strings of the form GPU: X with no difference in load. 
    #     They should uniquely be routed to individual gpus.

    #     An unknown prefix should get routed randomly to one of the gpus randomly(gpu selections all)
    #     """
    #     lp_tree_traversal = LPGurobiGreedyTraversal(num_gpus)
    #     texts = [
    #         f"GPU {i} This is a sample sentence on gpu {i}" for i in range(num_gpus)
    #     ]
    #     input_ids = self.tokenize_in_parallel(texts)
    #     cache = LPRadixCache()
    #     for i in range(num_gpus):
    #         lp_tree_traversal.insert_into_cache_and_solve(tree_cache=cache, input_ids=tuple(input_ids[i]), decode_cost=0.041)
    #         gpu_selections = cache.match_prefix_get_gpu_selection(input_ids[i])
    #     if verbose:
    #         lp_tree_traversal.pretty_print(cache.root_node)
    #     all_gpu_selections = set()
    #     for i in range(num_gpus):
    #         gpu_selections = cache.match_prefix_get_gpu_selection(input_ids[i])
    #         self.assertEqual(len(gpu_selections), 1)
    #         selected_gpu = gpu_selections.pop()
    #         self.assertFalse(selected_gpu in all_gpu_selections)
    #         all_gpu_selections.add(selected_gpu)
    #     # Test that if none of prefixes exist an empty set is returned
    #     text = "Unknown Prefix and text"
    #     unknown_input_id = self.tokenize_in_parallel([text])[0]
    #     node_map = lp_tree_traversal.node_map
    #     cache.insert(tuple(unknown_input_id), node_map=node_map)
    #     gpu_selection = cache.match_prefix_get_gpu_selection(unknown_input_id)
    #     self.assertEqual(gpu_selection, set(i for i in range(num_gpus)))

    def _test_basic_workload_branching_consistent_across_depths(self, data_size=3, verbose=True):
        lp_tree_traversal = LPGurobiGreedyTraversal(2)

        lorem5 = " ".join([self.lorem for _ in range(data_size)])
        texts = [
            f"Workload 1. {lorem5} A B C D",
            f"Workload 2. {lorem5} A B C D", 
            f"Workload 1. {lorem5} A B C D example 1",
            f"Workload 1. {lorem5} A B C D example 2",
            f"Workload 1. {lorem5} A B C D example 2 example 3",
            f"Workload 2. {lorem5} A B C D E",
            f"Workload 2. {lorem5} A B C D E F",
            f"Workload 2. {lorem5} A B C D E F G",
        ]
        input_ids = [self.tokenizer.encode(text) for text in texts]
        cache = LPRadixCache()
        runtime_selected_per_workload = {
            "Workload 1": [],
            "Workload 2": []
        }
        for i in range(len(texts)):
            node = lp_tree_traversal.insert_into_cache_and_solve(tree_cache=cache, input_ids=tuple(input_ids[i]), decode_cost=0.0988*45)
            gpu_selections = node.gpu_selections
            # lp_tree_traversal.pretty_print(cache.root_node, tokenizer=self.tokenizer)
            # print(f"GPU Selections: {gpu_selections}")
            self.assertEqual(len(gpu_selections), 1)
            if "Workload 1" in texts[i]:
                runtime_selected_per_workload["Workload 1"].append(list(gpu_selections)[0])
            else:
                runtime_selected_per_workload["Workload 2"].append(list(gpu_selections)[0])
            
        # Test that the same workloads get the same runtime
        self.assertEqual(len(set(runtime_selected_per_workload["Workload 1"])), 1)
        self.assertEqual(len(set(runtime_selected_per_workload["Workload 2"])), 1)
        # Check that the workloads are placed on different runtimes
        self.assertNotEqual(runtime_selected_per_workload["Workload 1"][0], runtime_selected_per_workload["Workload 2"][0])

    @parameterized.expand([
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
    ])
    def test_basic_workload_branching_consistent_across_depths(self, data_size):
        self._test_basic_workload_branching_consistent_across_depths(data_size)

    def test_replication_workload_branching_consistent_across_depths(self):
        lp_tree_traversal = LPGurobiGreedyTraversal(2)
        lorem5 = " ".join([self.lorem for _ in range(4)])
        texts = [
            f"Workload 1. {lorem5} A B C D",
            f"Workload 2. {lorem5} A B C D", 
        ]
        texts += [f"Workload 1. {lorem5} A B C D example" for i in range(1, 40)]
        input_ids = [self.tokenizer.encode(text) for text in texts]
        cache = LPRadixCache()
        workload_2_gpu_selections = set()
        workload_1_gpu_selections = set()
        for i in range(len(texts)):
            node = lp_tree_traversal.insert_into_cache_and_solve(tree_cache=cache, input_ids=tuple(input_ids[i]), decode_cost=0.0988*45)
            gpu_selections = node.gpu_selections
            if "Workload 1" in texts[i]:
                workload_1_gpu_selections.update(gpu_selections)
            else:
                self.assertEqual(len(gpu_selections), 1)
                workload_2_gpu_selections.add(list(gpu_selections)[0])

        self.assertEqual(len(workload_1_gpu_selections), 2)
        lp_tree_traversal.pretty_print(cache.root_node, tokenizer=self.tokenizer)

        # Test that workload 1 
        # lp_tree_traversal.pretty_print(cache.root_node, tokenizer=self.tokenizer)

def run_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner, exit=False)

    # with open('test_output.log', 'a+') as f:
    #     runner = unittest.TextTestRunner(stream=f, verbosity=2)
    #     unittest.main(testRunner=runner, exit=False)
if __name__ == "__main__":
    # test = TestLPTreeTraversal()
    # test.setUp()
    # test._test_lptree_traversal_basic_N_node(5)
    run_tests()