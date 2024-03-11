import hashlib
from consistent_hash_router import ConsistentHashingWithRadixCache
import random
from enum import Enum, auto
from typing import List, Optional
from collections import Callable
from dataclasses import dataclass
random.seed(10)

@dataclass
class CustomRuntimeSelector:
    """
    Provide a input function to hash the input text.
    Can be used for testing purposes in order to deterministcally send values via an orcale
    """
    InputText = str
    NodeID = int

    num_nodes: int
    def runtime_selector(self, text: InputText) -> NodeID:
        pass

class DataParallelRuntimeSelectionPolicy(Enum):
    RANDOM = auto()
    CONSISTENT_HASH = auto()
    CUSTOM = auto()


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

    def select_runtime(self, text, experiment_id, request_id) -> str:
        # TODO provide runtimes as model_id instead of just an int
        if self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.RANDOM:
            selected_runtime = random.randint(0, self.total_nodes - 1)
        elif (
            self.runtime_selection_policy
            == DataParallelRuntimeSelectionPolicy.CONSISTENT_HASH
        ):
            # prefix cache -> consistent hash select which runtime
            selected_runtime = self.consistent_radix_hash.get_node_for_key(text)
        elif self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.CUSTOM and self.custom_selector:
            selected_runtime = self.custom_selector.runtime_selector(text)
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
