import random
from enum import Enum, auto
from typing import List, Optional
from dataclasses import dataclass
random.seed(10)
import pandas as pd
from sglang.srt.managers.router.model_runner import GPUConfig
import threading

@dataclass
class CustomRuntimeSelector:
    """
    Provide a input function to hash the input text.
    Can be used for testing purposes in order to deterministcally send values via an orcale
    """
    InputText = str
    NodeID = int

    num_nodes: int
    def runtime_selector(self, text: InputText, request_id: str, input_ids: List, sampling_params) -> NodeID:
        pass

    def finish_request(self, text: InputText, request_id: str, input_ids: List, func_output) -> NodeID:
        pass

class DataParallelRuntimeSelectionPolicy(Enum):
    RANDOM = auto()
    ROUND_ROBIN = auto()

    CUSTOM = auto()

class CustomPolicyType(Enum):
    ORACLE = auto()
    ORACLE_HOT_COLD = auto()

    TBORACLE = auto()
    TBORACLE_B = auto()
    TB_DOMAIN_ORACLE = auto()

    LPM = auto()
    GLPM = auto()

    LOOGLE_ORACLE = auto()

    GREEDY_LP = auto()
    GREEDY_LP_OLD = auto()

    BASIC_MEM_SCHEDULER = auto()
    BASIC_MEM_SCHEDULERV2 = auto()
    BASIC_MEM_SCHEDULERV2_5 = auto()
    BasicMemSchedulerV3 = auto()

    HistogramBasedMemoryLoadScheduler = auto()
    HiostgramBasedRecompLoad = auto()
    HiostgramBasedRecompLoadWithEvictionV2 = auto()
    HiostgramBasedRecompLoadWithOutEvictionV2 = auto()

    MemSchedulerEvictBasedOnLoad = auto()
    MemSchedulerWithGlobalEviction = auto()

class DataParallelRequestRouter:
    def __init__(
        self,
        runtime_selection_policy: DataParallelRuntimeSelectionPolicy,
        total_nodes=2,
        custom_runtime_selector=None
    ):
        self.runtime_selection_policy = runtime_selection_policy
        self.custom_selector: Optional[CustomRuntimeSelector] = custom_runtime_selector
        self.total_nodes = total_nodes
        self.model_selection_stats = []
        self.lock = threading.Lock()
        self.counter = 0

    def select_runtime(self, text, experiment_id, request_id, input_ids=None, sampling_params=None) -> int:
        if self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.RANDOM:
            selected_runtime = random.randint(0, self.total_nodes - 1)
        elif self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.ROUND_ROBIN:
            with self.lock:
                selected_runtime = self.counter % self.total_nodes
                self.counter += 1
        elif self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.CUSTOM and self.custom_selector:
            selected_runtime = self.custom_selector.runtime_selector(text, request_id, input_ids, sampling_params)
        else:
            raise NotImplementedError(f"Runtime selection policy {self.runtime_selection_policy} not implemented with {self.custom_selector}")
        self.model_selection_stats.append(
            {
                "selected_runtime": selected_runtime,
                "text": text,
                "policy": self.runtime_selection_policy.name,
                "experiment_id": experiment_id,
                "request_id": request_id,
            }
        )
        return selected_runtime

    def finish_request(self, text, experiment_id, request_id, input_ids=None, func_output=None) -> int:
        if self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.RANDOM or self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.ROUND_ROBIN:
            pass
        elif self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.CUSTOM and self.custom_selector:
            self.custom_selector.finish_request(text, request_id, input_ids, func_output)
        else:
            raise NotImplementedError

    def update_runtime_selection_policy(self, runtime_selection_policy):
        self.runtime_selection_policy = runtime_selection_policy

    def get_model_selection_counts(self):
        df = pd.DataFrame(self.model_selection_stats)
        df.drop("text", axis=1, inplace=True)
        counts = df["selected_runtime"].value_counts().to_dict()
        return counts
