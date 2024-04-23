import unittest
from multi_node_loader import MultiNodeLoader
from sglang.srt.managers.router.model_runner import GPUConfig
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy
import asyncio
import requests
import time
import os
from huggingface_hub import login

if os.environ.get('HUGGINGFACE_TOKEN'):
    login(token=os.environ.get('HUGGINGFACE_TOKEN'))

class TestModelLoadingAndExecution(unittest.TestCase):

    def read_prompts(self, path):
        with open(path, 'r') as f:
            return f.read()
        
    def setUp(self) -> None:
        self.loaders = []
        self.model_details = []

    def test_load_and_run_sglang_runtime(self):

        gpu_configs = [
            GPUConfig(0, use_ssh=False)
        ]
        loader = MultiNodeLoader()
        self.loaders.append(loader)
        model_details = loader.load_model(
            model_path="mistralai/Mistral-7B-v0.1",
            gpu_configs=gpu_configs,
            log_prefix_hit=True,
            mem_fraction_static=0.8,
            context_length=65536,
        )
        self.model_details.append(model_details)
        model_details.update_runtime_selection_policy(DataParallelRuntimeSelectionPolicy.RANDOM, "")
        model_details.current_experiment_state_time = time.time()
        model_details.start_time = time.time()
        # prompt = self.read_prompts('multi_node/benchmarks/text.txt')
        prompt = "Hello, how are you? I am"
        # print(prompt)
        output = asyncio.run(model_details.async_send_request(
            text=prompt,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": 10,
            },
            input_ids=[0] * 10,
            rid='982734',
        ))
        print(output)

        # payload = {
        #     # "text": "Hello, how are you? I am",
        #     "text": prompt,
        #     "sampling_params": {
        #         "temperature": 0.0,
        #         "max_new_tokens": 10,
        #     },
        #     # "model": "Qwen/Qwen1.5-7B-Chat",
        #     "rid": '982734',
        #     "stream": True,
        #     # "model": "llama2"
        # }
        # print(model_details.runtimes[0].generate_url)
        # response = requests.post(model_details.runtimes[0].generate_url, json=payload, stream=True)
        # # print(response.json())
        # for chunk in response.iter_content(chunk_size=None):
        #     print(chunk.decode("utf-8"))

    def test_load_and_run_vllm_runtime(self):

        gpu_configs = [
            GPUConfig(0, use_ssh=False, ssh_config={
                "hostname": "192.168.1.16",
                "username": "dongming",
                "port": 456,
                "python_process": "/mnt/data/ssd/dongming/vllm_env/bin/python",
                "password": os.environ.get('SSH_PASSWORD')
            }, vllm_config={'vllm_port': 8080})
        ]
        loader = MultiNodeLoader()
        self.loaders.append(loader)
        model_details = loader.load_model(
            # model_path="Qwen/Qwen1.5-7B-Chat",
            model_path="mistralai/Mistral-7B-v0.1",
            gpu_configs=gpu_configs,
            log_prefix_hit=True,
            mem_fraction_static=0.8,
            context_length=65536,
            enable_prefix_caching=True,
        )
        self.model_details.append(model_details)
        model_details.update_runtime_selection_policy(DataParallelRuntimeSelectionPolicy.RANDOM, "")

        model_details.current_experiment_state_time = time.time()
        model_details.start_time = time.time()

        # prompt = self.read_prompts('multi_node/benchmarks/text.txt')
        prompt = "Hello, how are you? I am"
        # print(prompt)
        output = asyncio.run(model_details.async_send_request(
            text=prompt,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": 10,
            },
            input_ids=[0] * 10,
            rid='982734',
        ))
        print(output)

        # payload = {
        #     "prompt": prompt,
        #     "sampling_params": {
        #         "temperature": 0.0,
        #         "max_tokens": 10,
        #     },
        #     "model": "mistralai/Mistral-7B-v0.1",
        #     "rid": '982734',
        #     "stream": True,
        #     # "model": "llama2"
        # }
        # print(model_details.runtimes[0].generate_url)
        # response = requests.post(model_details.runtimes[0].generate_url, json=payload, stream=True)
        # # print(response.json())
        # print(response.status_code, response.reason)
        # for chunk in response.iter_content(chunk_size=None):
        #     print(chunk.decode("utf-8"))

    def tearDown(self) -> None:
        for loader, model_details in zip(self.loaders, self.model_details):
            loader.unload_model(model_details)


if __name__ == "__main__":
    unittest.main()