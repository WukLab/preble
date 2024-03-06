import hashlib
from consistent_hash_router import ConsistentHashingWithRadixCache
import random
from enum import Enum
from typing import List


class DataParallelRuntimeSelectionPolicy(Enum):
    RANDOM = 1
    RADIX_CACHE = 2


class DataParallelRequestRouter:
    def __init__(
        self,
        runtime_selection_policy: DataParallelRuntimeSelectionPolicy,
        total_nodes=2,
    ):
        self.runtime_selection_policy = runtime_selection_policy
        self.consistent_radix_hash = ConsistentHashingWithRadixCache(
            num_nodes=total_nodes
        )
        self.total_nodes = total_nodes
        self.model_selection_stats = []

    def select_runtime(self, text, experiment_id, request_id) -> str:
        # TODO provide runtimes as model_id instead of just an int
        if self.runtime_selection_policy == DataParallelRuntimeSelectionPolicy.RANDOM:
            selected_runtime = random.randint(0, self.total_nodes - 1)
        elif (
            self.runtime_selection_policy
            == DataParallelRuntimeSelectionPolicy.RADIX_CACHE
        ):
            # prefix cache -> consistent hash select which runtime
            selected_runtime = self.consistent_radix_hash.get_node_for_key(text)
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
