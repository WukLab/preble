from data_parallel_request_cache import (
    CustomRuntimeSelector,
)
from typing import List
import random
from dataclasses import dataclass
import re

@dataclass
class Oracle(CustomRuntimeSelector):
    num_workloads: int
    trace = {}

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None):
        num_nodes = self.num_nodes
        self.trace[request_id] = text[:50]
        for i in range(self.num_workloads):
            if text.startswith(f"Workload {i} "):
                return i % num_nodes

        return random.randint(0, num_nodes - 1)


@dataclass
class TBOracle:
    trace = {}
    tbl = {}
    num_nodes: int
    counter = {}

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None):
        match = re.search(r"You have access of the following tools:\n1.(.+?): ", text)
        if match:
            tool = match.group(1)
            self.counter[tool] = self.counter.get(tool, 0) + 1
            num_nodes = self.num_nodes
            if tool not in self.tbl:
                self.tbl[tool] = random.randint(0, num_nodes - 1)
            return self.tbl[tool]
        else:
            return random.randint(0, self.num_nodes - 1)


@dataclass
class TBOracleB(CustomRuntimeSelector):
    trace = {}
    tbl = {}
    counter: int = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None):
        match = re.search(r"You have access of the following tools:\n1.(.+?): ", text)
        if match:
            tool = match.group(1)
            if tool not in self.tbl:
                self.tbl[tool] = self.counter % self.num_nodes
                self.counter += 1
            return self.tbl[tool]
        else:
            return random.randint(0, self.num_nodes - 1)

@dataclass
class LoogleOracle(CustomRuntimeSelector):
    trace = {}
    tbl = {}
    counter: int = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None):
        match = re.search(r'(.*)Question:', text, re.DOTALL)
        if match:
            tool = match.group(1)
            if tool not in self.tbl:
                self.tbl[tool] = self.counter % self.num_nodes
                self.counter += 1
            return self.tbl[tool]
        else:
            return random.randint(0, self.num_nodes - 1)
