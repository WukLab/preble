import random
import warnings
import os
import sys
import numpy as np

random.seed(10)
np.random.seed(10)

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import datetime
import time
import multi_exp_configs.e2e_loogle_config

warnings.simplefilter(action="ignore", category=FutureWarning)

from multi_node_loader import MultiNodeLoader
from model_runtime_manager import RequestFuncOutput, ModelDetails
from benchmark_utils import BenchmarkMetrics, MajorExperimentArgs, WorkloadConfig
from benchmark_workload_gen import *
from data_parallel_request_cache import (
    DataParallelRuntimeSelectionPolicy,
    CustomPolicyType,
)
import gc
import logging
from typing import List
import torch
from data_parallel_request_cache import (
    CustomPolicyType,
    DataParallelRuntimeSelectionPolicy,
)
from greedy_lp import GurobiGreedyLPScheduler
from model_runtime_manager import ModelDetails, RequestFuncOutput
from multi_node_loader import MultiNodeLoader

from benchmark_utils import BenchmarkMetrics
from benchmark_workload_gen import *
from basic_mem_scheduler import BasicMemSchedulerV2
from multi_node.global_scheduler import GlobalScheduler
from multi_node.global_scheduler_with_time import GlobalSchedulerWithTime
from multi_experiment_benchmark_utils import DefaultWorkload, ConfigurableMajorExperimentArgs, AllExperiments, ExperimentType, Workload

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)
import warnings

# Ignore warnings
warnings.filterwarnings("ignore", message="huggingface/tokenizers")
log = logging.getLogger(__name__)


def register_selector(
    model_details: ModelDetails, workload_config: Workload
):
    policy, custom_policy = workload_config.policy, workload_config.custom_policy
    if policy != DataParallelRuntimeSelectionPolicy.CUSTOM:
        model_details.update_runtime_selection_policy(policy, None)
        return

    # Handling for specific Oracle types that share similar instantiation patterns
    oracle_creators = {
        CustomPolicyType.ORACLE: Oracle,
        CustomPolicyType.ORACLE_HOT_COLD: OracleHotCold,
        CustomPolicyType.TBORACLE: TBOracle,
        CustomPolicyType.TBORACLE_B: TBOracleB,
        CustomPolicyType.LOOGLE_ORACLE: LoogleOracle,
        CustomPolicyType.TB_DOMAIN_ORACLE: TBMultiDomainOracle,
    }

    def handle_oracle(oracle_type):
        """Generic handler for oracle types."""
        if oracle_type in (CustomPolicyType.ORACLE, CustomPolicyType.ORACLE_HOT_COLD):
            return oracle_creators[oracle_type](
                num_nodes=len(model_details.runtimes),
                num_workloads=workload_config.num_prefix_patterns,
            )
        else:
            return oracle_creators[oracle_type](num_nodes=len(model_details.runtimes))

    if custom_policy in oracle_creators:
        custom_runtime_selector = handle_oracle(custom_policy)
        model_details.update_runtime_selection_policy(
            DataParallelRuntimeSelectionPolicy.CUSTOM,
            custom_runtime_selector=custom_runtime_selector,
        )
        return

    def handle_basic_mem_schedulerv2():
        return BasicMemSchedulerV2(num_nodes=len(model_details.runtimes))

    def handle_histogram_based_recomp():
        return GlobalScheduler(
            num_nodes=len(model_details.runtimes), enable_eviction=False, enable_miss_rate=True
        )

    def handle_histogram_based_recomp_without_miss_rate():
        return GlobalScheduler(
            num_nodes=len(model_details.runtimes), enable_eviction=False, enable_miss_rate=False
        )
    
    selector_creators = {
        CustomPolicyType.BASIC_MEM_SCHEDULERV2: handle_basic_mem_schedulerv2,
        CustomPolicyType.GlobalScheduler: handle_histogram_based_recomp,
        CustomPolicyType.GlobalSchedulerWithoutMissRate: handle_histogram_based_recomp_without_miss_rate,
        CustomPolicyType.GlobalSchedulerTime: lambda: GlobalSchedulerWithTime(num_nodes=len(model_details.runtimes)),
        CustomPolicyType.GlobalSchedulerTimeWithEviction: lambda: GlobalSchedulerWithTime(num_nodes=len(model_details.runtimes), enable_eviction=True),
    }

    if custom_policy not in selector_creators:
        logging.error(f"Custom policy {custom_policy} not found.")
        return

    creator = selector_creators[custom_policy]
    custom_runtime_selector = creator()
    model_details.update_runtime_selection_policy(
        DataParallelRuntimeSelectionPolicy.CUSTOM,
        custom_runtime_selector=custom_runtime_selector,
    )


def run_single_workload(
    model_details: ModelDetails,
    workload_config: Workload,
    experiment_type: ExperimentType,
    csv_file: str,
    trace_json_file: Optional[str]
):
    logging.info(
        workload_config.get_starting_policy_message()
    )
    register_selector(
        model_details, workload_config
    )

    torch.manual_seed(10)
    np.random.seed(10)

    tic_benchmark = time.time()
    results: List[RequestFuncOutput] = model_details.get_experiment_results_for_experiment_type(
        workload_config,
        experiment_type=experiment_type,
    )
    overall_latency = time.time() - tic_benchmark

    counts = model_details.request_router.get_model_selection_counts()
    exp_params = workload_config.workload_params_str()
    bench_metrics = BenchmarkMetrics.gen_benchmark_metrics(
        tokenizer=workload_config.get_tokenizer(),
        req_func_outputs=results,
        overall_latency=overall_latency,
        time_limit=workload_config.exp_time,
        gpu_counts=counts,
        detail_log_path=trace_json_file,
    )
    bench_metrics.to_log_file(exp_params)
    bench_metrics.to_csv_file(csv_file, exp_params)

def run_experiment(configurable_exp_args: ConfigurableMajorExperimentArgs):
    loader = MultiNodeLoader(configurable_exp_args.simulate)
    
    directory = os.path.dirname(configurable_exp_args.log_file_path)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    logging.basicConfig(level=logging.INFO, filename=configurable_exp_args.log_file_path)
    formatter = logging.Formatter(
        "%(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    # Add current time to log file
    start_date = datetime.datetime.utcnow()
    start_time = time.time()
    logging.info(f"Starting Experiment at {start_date} with {configurable_exp_args.experiment_name}")
    for workload_config in configurable_exp_args.workload_configs:
        model_details = loader.load_model(
            model_path=configurable_exp_args.model_path, gpu_configs=workload_config.server_configs
        )
        trace_json_file = directory + '/' + workload_config.workload_params_str().replace(", ", "_").replace("/", "-") + '.json'
        run_single_workload(
            model_details,
            workload_config,
            csv_file=configurable_exp_args.csv_log_path,
            trace_json_file=trace_json_file,
            experiment_type=configurable_exp_args.experiment_type,
        )
        loader.unload_model(model_details)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(10)
    logging.info(f"Total Experiment Time: {time.time() - start_time}")


def run_all_experiments(all_experiments: AllExperiments):
    for experiment in all_experiments.experiments:
        run_experiment(experiment)

if __name__ == "__main__":
    # from multi_node.benchmarks.multi_exp_configs.e2e_4r_toolbench_config import toolbench_experiment as tb_4r_config
    # from multi_node.benchmarks.multi_exp_configs.e2e_2r_toolbench_config import toolbench_experiment as tb_2r_config
    # exp_args = AllExperiments([tb_2r_config, tb_4r_config])
    
    # from multi_node.benchmarks.multi_exp_configs.e2e_2r_loogle_config import loogle_experiment as lg_2r_config
    # from multi_node.benchmarks.multi_exp_configs.e2e_4r_loogle_config import loogle_experiment as lg_4r_config
    # exp_args = AllExperiments([lg_2r_config, lg_4r_config])
    
    # from multi_node.benchmarks.multi_exp_configs.e2e_videoQA_config import videoQA_experiment as vqa_2r_config
    # from multi_node.benchmarks.multi_exp_configs.e2e_4r_videoQA_config import videoQA_experiment as vqa_4r_config
    # exp_args = AllExperiments([vqa_2r_config, vqa_4r_config]) 
    
    
    # from benchmarks.multi_exp_configs.e2e_loogle_config import exp_args
#     from benchmarks.multi_exp_configs.e2e_loogle_config import exp_args
    # from benchmarks.multi_exp_configs.e2e_mix_config import exp_args
    # from benchmarks.multi_exp_configs.e2e_videoQA_config import exp_args
    # from benchmarks.multi_exp_configs.e2e_toolbench_config import exp_args
    # from benchmarks.multi_exp_configs.e2e_virtualenv_config import exp_args
    from multi_node.benchmarks.multi_exp_configs.e2e_234r_loogle_config import exp_args
    # from multi_node.benchmarks.multi_exp_configs.e2e_234r_toolbench_config import exp_args
    
    run_all_experiments(exp_args)
