from dataclasses import dataclass
from typing import List, Dict
from model_runtime_manager import RequestFuncOutput
import numpy as np
import logging

@dataclass
class BenchmarkMetrics:
    num_finished_requests: int
    average_finished_topt: float
    ttfts: List[float]
    tpots: List[float]
    throughput_tok_sec: float
    all_results: List[RequestFuncOutput]
    average_request_latency: float
    std_request_latency: float
    average_p90: float
    max_latency: float
    p99_latency: float
    average_ttft: float
    average_topt: float
    prefill_decode_ratio: float
    overall_latency: float
    requests_per_sec: float
    gpu_counts: Dict[int, int]

    def gen_benchmark_metrics(tokenizer, 
                              req_func_outputs: List[RequestFuncOutput], 
                              overall_latency:float, 
                              time_limit: int = 100,
                              gpu_counts={}):
        for result in req_func_outputs:
            result.update_metrics(tokenizer) # Computes the generated output tokens
        
        ttfts = [result.ttft for result in req_func_outputs]
        tpots = [result.tpot for result in req_func_outputs if result.tpot is not None]
        overall_latency = overall_latency
        request_latencies = [result.request_latency for result in req_func_outputs]
        throughput_tok_sec = sum([result.total_tokens for result in req_func_outputs]) / overall_latency
        all_results = req_func_outputs

        num_finished_requests = sum(
            [result.global_time <= time_limit for result in req_func_outputs if result.success]
        )
        average_finished_tpot =  np.average([
            result.tpot for result in req_func_outputs if result.tpot is not None and result.global_time <= time_limit and result.success
        ])
        prefill_decode_ratio = np.average([
            result.prefill_decode_ratio for result in req_func_outputs
        ])

        average_request_latency, std_request_latency, average_p90 = (
            np.mean(request_latencies),
            np.std(request_latencies),
            np.percentile(request_latencies, 90),
        )
        max_latency, p99_latency = np.max(request_latencies), np.percentile(
            request_latencies, 99
        )
        average_ttft = np.mean(ttfts)
        average_topt = np.mean(tpots)
        requests_per_sec = len(req_func_outputs) / overall_latency
        return BenchmarkMetrics(
            num_finished_requests=num_finished_requests,
            average_finished_topt=average_finished_tpot,

            ttfts=ttfts,
            tpots=tpots,
            throughput_tok_sec=throughput_tok_sec,
            all_results=all_results,
            average_request_latency=average_request_latency,
            overall_latency=overall_latency,
            std_request_latency=std_request_latency,

            average_p90=average_p90,
            max_latency=max_latency,
            p99_latency=p99_latency,

            average_ttft=average_ttft,
            average_topt=average_topt,

            prefill_decode_ratio=prefill_decode_ratio,
            requests_per_sec=requests_per_sec,
            gpu_counts = gpu_counts
        )
    
    def to_json(self):
        all_reqs = [result.to_json() for result in self.all_results]
        return {
            "num_finished_requests": self.num_finished_requests,
            "average_finished_topt": self.average_finished_topt,
            "ttfts": self.ttfts,
            "tpots": self.tpots,
            "overall_latency": self.overall_latency,
            "average_request_latency": self.average_request_latency,
            "std_request_latency": self.std_request_latency,
            "average_p90": self.average_p90,
            "max_latency": self.max_latency,
            "p99_latency": self.p99_latency,
            "average_ttft": self.average_ttft,
            "average_topt": self.average_topt,
            "throughput_tok_sec": self.throughput_tok_sec,
            "all_reqs": all_reqs,
            "prefill_decode_ratio": self.prefill_decode_ratio
        }

    def to_log_file(self, exp_params):
        logging.debug(
            f"Params=({exp_params}) Overall Latency: {self.overall_latency}"
        )
        logging.debug(
            f"Params=({exp_params}) Overall Throughput: {self.requests_per_sec}"
        )
        logging.debug(
            f"Params=({exp_params}) Overall Request Latency: {self.average_request_latency}, STD: {self.std_request_latency}, P90: {self.average_p90}"
        )
        logging.debug(
            f"Params=({exp_params}) Average TTFT: {self.average_ttft}, Average TOPT: {self.average_topt}, Throughput ToksPerSec: {self.throughput_tok_sec}"
        )
        logging.debug(
            f"Params=({exp_params}) Num Finished Requests: {self.num_finished_requests}, Finished Throughput ToksPerSec: {self.average_finished_topt}"
        )
        logging.debug(
            f"Params=({exp_params}) Overall Max Latency: {self.max_latency}, P99: {self.p99_latency}"
        )
        logging.debug(
            f"Params=({exp_params}) Overall PrefillRatio: {self.prefill_decode_ratio}"
        )
        logging.debug(
            f"Params=({exp_params}) Counts: {self.gpu_counts}"
        )
