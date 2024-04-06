from consistent_hash_router import ConsistentHashingWithRadixCache
import random
from enum import Enum, auto
from typing import List, Optional
from dataclasses import dataclass
random.seed(10)
import pandas as pd
from sglang.srt.managers.router.model_runner import GPUConfig

@dataclass
class CustomRuntimeSelector:
    """
    Provide a input function to hash the input text.
    Can be used for testing purposes in order to deterministcally send values via an orcale
    """
    InputText = str
    NodeID = int

    num_nodes: int
    def runtime_selector(self, text: InputText, request_id: str, input_ids: List) -> NodeID:
        pass
    
class DataParallelRuntimeSelectionPolicy(Enum):
    RANDOM = auto()
    CONSISTENT_HASH = auto()
    CUSTOM = auto()

class CustomPolicyType(Enum):
    ORACLE = auto()

    TBORACLE = auto()
    TBORACLE_B = auto()

    LPM = auto()
    GLPM = auto()

    LOOGLE_ORACLE = auto()

    LP_SCHEDULER = auto()

class DataParallelRequestRouter:
    def __init__(
        self,
        runtime_selection_policy: DataParallelRuntimeSelectionPolicy,
        total_nodes=2,
        custom_runtime_selector=None
    ):
        self.runtime_selection_policy = runtime_selection_policy
        self.consistent_radix_hash = ConsistentHashingWithRadixCache(
            num_nodes=total_nodes
        )
        self.custom_selector: Optional[CustomRuntimeSelector] = custom_runtime_selector
        self.total_nodes = total_nodes
        self.model_selection_stats = []

    def select_runtime(self, text, experiment_id, request_id, input_ids=None) -> int:
        if self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.RANDOM:
            selected_runtime = random.randint(0, self.total_nodes - 1)
        elif self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.CUSTOM and self.custom_selector:
            selected_runtime = self.custom_selector.runtime_selector(text, request_id, input_ids)
        else:
            raise NotImplementedError
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

    def update_runtime_selection_policy(self, runtime_selection_policy):
        self.runtime_selection_policy = runtime_selection_policy

    def get_model_selection_counts(self):
        df = pd.DataFrame(self.model_selection_stats)
        df.drop("text", axis=1, inplace=True)
        counts = df["selected_runtime"].value_counts().to_dict()
        return counts
