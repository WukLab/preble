# For each dataset, test that it loads and it can run for 8 iterations to a log file
from typing import List, Tuple
from gpu_stats_profiling import get_gpu_profile
from model_runtime_manager import ModelDetails
from multi_node_loader import MultiNodeLoader, GPUConfig
import time
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy
import random
import asyncio
import torch
import gc
from benchmarks.benchmark_utils import BenchmarkMetrics
from transformers import AutoTokenizer
from benchmarks.benchmark_workload_gen import RandomDataLoader, ToolBenchDataLoader, LoadDistribution, LooGLEDataset, LooGLEDatasetType
import unittest
from parameterized import parameterized
import logging

# disable huggingface/tokenizers: warning
logging.getLogger("transformers").setLevel(logging.ERROR)

def test_random_policy_on_dataset(
    requests,
    rps=0.0,
    exp_time=200,
    context_length=4096,
    gpu_configs=None,
    model_name="mistralai/Mistral-7B-v0.1",
) -> BenchmarkMetrics:
    random.seed(10)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    loader = MultiNodeLoader()
    model_details = loader.load_model(
        model_name,
        gpu_configs=gpu_configs,
        log_prefix_hit=True,
        mem_fraction_static=0.8,
        context_length=context_length,
    )

    model_details.update_runtime_selection_policy(DataParallelRuntimeSelectionPolicy.RANDOM)

    tic_benchmark = time.time()
    results = asyncio.run(
        model_details.async_generate_batch_request_per_sec(
            requests,
            rps,
            model_details.async_send_request,
            exp_time,
        )
    )
    overall_latency = time.time() - tic_benchmark
    counts = model_details.request_router.get_model_selection_counts()

    bench_metrics = BenchmarkMetrics.gen_benchmark_metrics(
        tokenizer=tokenizer,
        req_func_outputs=results,
        overall_latency=overall_latency,
        time_limit=100,
        gpu_counts=counts,
    )

    loader.unload_model(model_details)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
    return bench_metrics


class TestMultiNodeLoaderWithDatasets(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        # Logging filter to error only
        # logging.basicConfig(level=logging.ERROR)
        return super().setUp()

    def test_random_dataset(self):
        """
            SSH transport issues might occur on small tests. This tested from 1 machine 06 to 08
        """
        gpu_configs = [GPUConfig(gpu_id=0), GPUConfig(gpu_id=1)]
        total_requests = 20
        requests = RandomDataLoader(
            4, total_requests, tokenizer=self.tokenizer, random_workload_path="benchmarks/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
        ).generate_workload(k=1.1)
        benchmark_results = test_random_policy_on_dataset(
            requests,
            rps=8, 
            exp_time=200, 
            context_length=4096, 
            gpu_configs=gpu_configs
        )
        self.assertEqual(benchmark_results.num_sucessful_requests, total_requests)


    def test_toolbench_dataset(self):
        """
            SSH transport issues might occur on small tests. This tested from 1 machine 06 to 08
        """
        gpu_configs = [GPUConfig(gpu_id=0), GPUConfig(gpu_id=1)]
        total_requests = 200
        dataloader = ToolBenchDataLoader(
            "benchmarks/datasets/G1_workload_updated_with_input_output.json",
            100,
            total_requests,
            self.tokenizer,
            load_dist=LoadDistribution.EVEN,
        )
        requests = dataloader.generate_workload(k=1.1)
        benchmark_results = test_random_policy_on_dataset(
            requests,
            rps=8, 
            exp_time=float('inf'), 
            context_length=4096, 
            gpu_configs=gpu_configs
        )
        self.assertEqual(benchmark_results.num_sucessful_requests, total_requests)

    def test_loogle_dataset(self):
        gpu_configs = [GPUConfig(gpu_id=0), GPUConfig(gpu_id=1)]
        total_requests = 20

        dataloader_short = LooGLEDataset(
            loogle_dataset_type=LooGLEDatasetType.SHORT_QA,
            num_patterns=2,
            total_num_requests=total_requests,
            tokenizer=self.tokenizer,
            load_dist=LoadDistribution.ALL,
            crop_max_decode=True,
        )
        requests = dataloader_short.generate_workload(max_length=32768)
        total_requests = len(requests)
        benchmark_results = test_random_policy_on_dataset(
            requests,
            rps=1, 
            exp_time=float('inf'), 
            context_length=32768, 
            gpu_configs=gpu_configs
        )
        self.assertEqual(benchmark_results.num_sucessful_requests, total_requests)

    def get_ssh_gpu_config(self, gpu_id):
        return GPUConfig(
            gpu_id=gpu_id,
            url=None,
            use_ssh=True,
            ssh_config={
                    "hostname": "192.168.1.18",
                    "username": "vikranth",
                    "port": 456,
                    "python_process": "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python",
                    "node_name": "08",
                }
        )

    @parameterized.expand([
        (1,),
        (2,),
    ])
    def test_executing_model_over_ssh(self, num_gpus):
        self._test_executing_model_over_ssh_long(num_gpus)

    def _test_executing_model_over_ssh_long(self, num_gpus=1):
        """
            SSH transport issues might occur on small tests. This tested from 1 machine 06 to 08
        """
        gpu_configs = [self.get_ssh_gpu_config(i) for i in range(num_gpus)]
        total_requests = 200
        requests = RandomDataLoader(4, total_requests, tokenizer=self.tokenizer, random_workload_path="benchmarks/datasets/ShareGPT_V3_unfiltered_cleaned_split.json").generate_workload(k=1.1)
        benchmark_results = test_random_policy_on_dataset(
            requests,
            rps=8, 
            exp_time=200, 
            context_length=4096, 
            gpu_configs=gpu_configs
        )
        self.assertEqual(benchmark_results.num_sucessful_requests, total_requests)

    def test_mixing_ssh_nodes_with_current_gpu_nodes(self):
        """
            SSH transport issues might occur on small tests. This tested from 1 machine 06 to 08
        """
        gpu_configs = [self.get_ssh_gpu_config(i) for i in range(2)] + [GPUConfig(gpu_id=0)]
        total_requests = 200
        requests = RandomDataLoader(4, total_requests, tokenizer=self.tokenizer, random_workload_path="benchmarks/datasets/ShareGPT_V3_unfiltered_cleaned_split.json").generate_workload(k=1.1)
        benchmark_results = test_random_policy_on_dataset(
            requests,
            rps=8, 
            exp_time=200, 
            context_length=4096, 
            gpu_configs=gpu_configs
        )
        self.assertEqual(benchmark_results.num_sucessful_requests, total_requests)
def run_tests():
    with open('test_output.log', 'a+') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)
if __name__ == "__main__":
    # test_multi_node_loader = TestMultiNodeLoaderWithDatasets()
    # test_multi_node_loader.setUp()
    # test_multi_node_loader.test_loogle_dataset()
    # test_multi_node_loader.test_random_dataset()
    # with open('test_output.log', 'w') as f:
    #     runner = unittest.TextTestRunner(stream=f, verbosity=2)
    #     unittest.main(testRunner=runner, exit=False)
    run_tests()
