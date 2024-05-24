import sys
import os
import uuid

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpu_stats_profiling import Monitor

from sglang.srt.server import Runtime as SGLangServer
import requests
import concurrent.futures
import unittest
from benchmarks.benchmark_workload_gen import get_react_workload
from simulator import ServerRuntimeSimulator
from benchmarks.exp_configs.react_mixed_config import add_simulation_to_gpu_config
from sglang.srt.managers.router.model_runner import GPUConfig
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.managers.io_struct import (
    TokenizedGenerateReqInput
)

gpu_config = GPUConfig(gpu_id=0)
add_simulation_to_gpu_config([gpu_config])


class TestSGLangHitRatioInspect(unittest.TestCase):
    def test_sglang_server_hit_ratio(self):
        """
        End to End SgLang Server test to test hit ratio after sending multiple requests.
        """
        runtime = ServerRuntimeSimulator(
            model_path="mistralai/Mistral-7B-v0.1",
            context_length=33000,
            gpu_config=gpu_config,
        )
        tokenizer = runtime.model_rpc.tokenizer
        
        sampling_params = SamplingParams(max_new_tokens = 1)
        sampling_params.normalize(tokenizer)
        sampling_params.verify()
        
        pixel_values, image_hash, image_size = None, None, None
        tokenized_input = TokenizedGenerateReqInput(
            rid=uuid.uuid4().hex,
            input_text='',
            input_ids=[0]*100,
            pixel_values=pixel_values,
            image_hash=image_hash,
            image_size=image_size,
            sampling_params=sampling_params,
            return_logprob=False,
            logprob_start_len=False,
            stream=False,
            arrival_time=0,
        )
        
        runtime.simulate_step([tokenized_input], 0)
        print(f'hit_ratio: {runtime.model_rpc.get_hit_ratio()}')
        tokenized_input = TokenizedGenerateReqInput(
            rid=uuid.uuid4().hex,
            input_text='',
            input_ids=[0]*100,
            pixel_values=pixel_values,
            image_hash=image_hash,
            image_size=image_size,
            sampling_params=sampling_params,
            return_logprob=False,
            logprob_start_len=False,
            stream=False,
            arrival_time=0,
        )
        runtime.simulate_step([tokenized_input], 31) # first goes out of window
        print(f'hit_ratio: {runtime.model_rpc.get_hit_ratio()}')
        self.assertEqual(runtime.model_rpc.get_hit_ratio(), 0.99)
        
        tokenized_input = TokenizedGenerateReqInput(
            rid=uuid.uuid4().hex,
            input_text='',
            input_ids=[1]*100,
            pixel_values=pixel_values,
            image_hash=image_hash,
            image_size=image_size,
            sampling_params=sampling_params,
            return_logprob=False,
            logprob_start_len=False,
            stream=False,
            arrival_time=0,
        )
        runtime.simulate_step([tokenized_input], 32) # second stays in window
        print(f'hit_ratio: {runtime.model_rpc.get_hit_ratio()}')
        self.assertEqual(runtime.model_rpc.get_hit_ratio(), 0.495)
        

if __name__ == "__main__":
    unittest.main()
