import sys
import os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpu_stats_profiling import Monitor

from sglang.srt.server import Runtime as SGLangServer
import requests
import concurrent.futures
import unittest
from benchmarks.benchmark_workload_gen import get_react_workload

class TestSGLangServerMetrics(unittest.TestCase):
    def test_sglang_server_metrics_basic(self):
        """
        End to End SgLang Server test to test launching server then requesting metrics
        """
        server = SGLangServer(
            model_path="mistralai/Mistral-7B-v0.1",
            context_length=1024,
        )
        metrics_url = f"{server.url}/scheduling_metrics"
        generate_url = f"{server.url}/generate"

        r = requests.post(generate_url, json={ 
                        "text": "Say this is a warmup request.",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 16,
                        },
                        })
        metrics_results1 = requests.post(metrics_url, json={"prompt": "sample prompt"})
        metrics_data1 = metrics_results1.json()
        token_kv_available_size = metrics_data1['token_kv_available_size']
        tree_cache_metrics_hit = metrics_data1['tree_cache_metrics_hit']
        tree_cache_metrics_total = metrics_data1['tree_cache_metrics_total']
        prefix_match_len = metrics_data1['prefix_match_len']
        evicatable_size = metrics_data1['evicatable_size']
        input_len = metrics_data1['input_len']

        self.assertTrue(token_kv_available_size > 0)
        self.assertEqual(prefix_match_len, 1)
        self.assertEqual(evicatable_size, 24)
        self.assertEqual(input_len, 3)
    
        r = requests.post(generate_url, json={ 
                        "text": "Say this is different warmup request.",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 16,
                        },
                        })
        # print(r.json())
        r = requests.post(metrics_url, json={"prompt": "Say this is different warmup request"})
        metrics_data2 = r.json()

        self.assertEqual(metrics_data2['prefix_match_len'], 8)
        self.assertEqual(metrics_data1['token_kv_available_size'] - metrics_data2['token_kv_available_size'], metrics_data2['evicatable_size'] - metrics_data1['evicatable_size'])
        self.assertEqual(metrics_data2['evicatable_size'], 44)
        self.assertTrue(metrics_data2['tree_cache_metrics_hit'] > metrics_data1['tree_cache_metrics_hit'])
        self.assertEqual(metrics_data2['input_len'], 8)
        server.shutdown()


    def test_metrics_server_waiting_queue(self):
        """
        End to End SgLang Server test to test launching server then requesting metrics
        """
        server = SGLangServer(
            model_path="mistralai/Mistral-7B-v0.1",
            context_length=1024,
            mem_fraction_static=0.8
        )
        metrics_url = f"{server.url}/scheduling_metrics"
        generate_url = f"{server.url}/generate"

        def send_N_concurrent_requests(N=200):
            futures = []
            metrics_data = []
            def send_request(url, payload):
                response = requests.post(url, json=payload)
                return response.json()
            def send_request_metrics(url, payload):
                response = requests.post(url, json=payload)
                metrics_data.append(response.json())
                return response.json()
            with concurrent.futures.ThreadPoolExecutor(256) as executor:
                for i in range(N):
                    future = executor.submit(send_request, generate_url, {
                        "text": get_react_workload(f"Workload {i}"),
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 30,
                        },
                    })
                    futures.append(future)
                    if i % 24 == 0:
                        future = executor.submit(send_request_metrics, metrics_url, {
                            "prompt": get_react_workload(f"Workload {i%10}"),
                        })
                        futures.append(future)
            # Send the metrics url request
            # Wait for all requests to complete
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
            return metrics_data
        # Create a list to store the futures
        metrics_data = send_N_concurrent_requests(256)
        # max_running_req = max(data['running_req_len'] for data in metrics_data)
        max_waiting_queue = max(data['waiting_queue_len'] for data in metrics_data)
        self.assertTrue(max_waiting_queue > 0)
        # Check the size of waiting queue is not 0
        
        server.shutdown()

if __name__ == "__main__":
    unittest.main()
