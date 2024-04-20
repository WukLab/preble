import random
import json
import string
import uuid

import numpy as np
import random
from enum import Enum, auto
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing import List, Optional, Tuple, Union
import math
import copy
from collections import defaultdict
import scipy.stats as ss
from data_parallel_request_cache import (
    CustomRuntimeSelector,
)
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import logging
from datasets import load_dataset
import re

ReActWorkloadEx1 = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]
"""

ReActWorkloadEx2 = """
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]
"""
ReActWorkloadEx3 = """
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]
"""

ReActWorkloadEx4 = """
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]
"""

ReActWorkload = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


def gen_random_string(length=100):
    cha_set = string.ascii_letters + string.digits
    ret = "".join(random.choices(cha_set, k=length))
    return ret


def get_react_workload(new_prefix, num_examples=4, num_new_chars=100):
    # Return workload 1 with a randoms string at the end
    examples = [ReActWorkloadEx1, ReActWorkloadEx2, ReActWorkloadEx3, ReActWorkloadEx4]
    # Add examples to ReAct Workload
    workload_prompt = ReActWorkload
    for i in range(num_examples):
        # if i >= len(examples):
        #     workload_prompt += random.choice(examples) + "\n"
        # else:
        j = i % len(examples)
        workload_prompt += examples[j] + "\n"
    return new_prefix + workload_prompt + gen_random_string(num_new_chars)


share_gpt_dataset = None


# Add ShareGPTDataset
def generate_random_workload(random_workload_path):
    global share_gpt_dataset
    if not share_gpt_dataset:
        if not random_workload_path:
            random_workload_path = "datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
        with open(random_workload_path) as f:
            data = json.load(f)
        share_gpt_dataset = data

        pts = []
    prompts = []
    for i in range(100):
        conv01 = share_gpt_dataset[i]
        conversations = conv01["conversations"]
        for conversation in conversations:
            if conversation["from"] == "human":
                prompts.append(conversation["value"])
                # share_gpt_workload.append({
                #     "prompt": conversation["value"],
                #     "gpt": None
                # })
                # TODO Currently just getting a list of prompts. Not considering chains of calls. Can probably run through SGLang frontend
                break
    return prompts
    # print(sum(avg)/len(avg))


class LoadDistribution(Enum):
    EVEN = auto()
    ALL = auto()
    ZIPF = auto()
    NORMAL = auto()


class DataLoader:
    def __init__(
        self,
        data_path: str,
        num_patterns: int,
        total_num_requests: int,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        load_dist: LoadDistribution = LoadDistribution.EVEN,
    ):
        self.data_path = data_path
        self.num_patterns = num_patterns
        self.total_num_requests = total_num_requests
        self.tokenizer = tokenizer
        self.load_dist = load_dist

    def generate_workload(self):
        raise NotImplementedError()
    
    def get_token_ids(self, request, tokenizer):
        input_ids = tokenizer(request["text"]).input_ids
        request["input_ids"] = input_ids

    def add_input_token_ids_to_workload(self, workload):
        with ThreadPoolExecutor(64) as executor:
            futures = []
            for request in workload:
                futures.append(executor.submit(self.get_token_ids, request, self.tokenizer))
            for future in futures:
                future.result()

class WorkloadPrefixDataLoader(DataLoader):
    def __init__(
        self,
        num_patterns: int,
        total_num_requests: int,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        load_dist: LoadDistribution = LoadDistribution.EVEN,
        distribution_of_non_shared: float = 0.0,
        output_len: int = 1,
        num_in_context_examples: int = 4,
        random_workload_path=None
    ):
        super().__init__(
            "random", num_patterns, total_num_requests, tokenizer, load_dist
        )
        self.distribution_of_non_shared = distribution_of_non_shared
        self.output_len = output_len
        self.num_in_context_examples = num_in_context_examples
        self.random_workload_path=random_workload_path

    def generate_workload(self, k):
        num_prefixed_shared = int(
            self.total_num_requests * (1 - self.distribution_of_non_shared)
        )
        num_non_shared = int(self.total_num_requests * self.distribution_of_non_shared)
        workload = []
        sampling_params = {
            "experiment_id": f"random_experiment_{self.num_patterns}_{self.distribution_of_non_shared}_{self.total_num_requests}",
            "temperature": 0,
            "max_new_tokens": self.output_len,
            "ignore_eos": True, # For better micro-benchmark
        }
        for i in range(num_prefixed_shared):
            workload_num = i % self.num_patterns
            prompt = get_react_workload(
                f"Workload {workload_num} ", num_examples=self.num_in_context_examples
            )
            workload.append(
                {
                    "text": prompt,
                    "sampling_params": copy.deepcopy(sampling_params),
                    "rid": uuid.uuid4().hex,
                }
            )
        # random_workload = generate_random_workload(random_workload_path=self.random_workload_path)
        for _ in range(num_non_shared):
            # prompt = random.choice(random_workload)
            prompt = get_react_workload(
                uuid.uuid4().hex + " ", num_examples=self.num_in_context_examples
            )
            workload.append(
                {
                    "text": prompt,
                    "sampling_params": copy.deepcopy(sampling_params),
                    "rid": uuid.uuid4().hex,
                }
            )
        self.add_input_token_ids_to_workload(workload)
        random.shuffle(workload)

        prompt_lens = [len(p["input_ids"]) for p in workload]
        plt.hist(prompt_lens)
        plt.savefig(f"react_prompt_length.png")
        return workload
    
    @staticmethod
    def is_hot(output):
        return output.prompt_text.startswith("Workload ")

class ToolBenchDataLoader(DataLoader):
    def __init__(
        self,
        data_path: str,
        num_patterns: int,
        total_num_requests: int,
        tokenizer,
        load_dist: LoadDistribution = LoadDistribution.EVEN,
    ):
        super().__init__(
            data_path, num_patterns, total_num_requests, tokenizer, load_dist
        )
        self.data = self.read_data()

    def read_data(self):
        data = json.load(open(self.data_path, "r"))
        return data

    def generate_workload(self, k=None):
        workload = []
        if self.load_dist == LoadDistribution.EVEN:
            load_threshold = math.ceil(self.total_num_requests // self.num_patterns)
            prefix_stats = [p for p, l in self.data.items() if len(l) >= load_threshold]
            selected_prefixs = np.random.choice(
                prefix_stats, self.num_patterns, replace=True
            )
            for p in selected_prefixs:
                selected_instances = np.random.choice(
                    self.data[p], load_threshold, replace=True
                )
                for e in selected_instances:
                    output_len = len(self.tokenizer(e["output"]).input_ids)
                    workload.append(
                        {
                            "text": e['prompt'], 
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": output_len,
                            },
                        }
                    )
        elif self.load_dist == LoadDistribution.ZIPF:
            assert k is not None
            prefix_stats = sorted(
                [(p, len(l)) for p, l in self.data.items()],
                key=lambda x: x[1],
                reverse=True,
            )[: self.num_patterns]
            # ZIPF distribution
            # sample hit to each selected prefix with the given distribution
            hist = []
            while len(hist) < self.total_num_requests:
                tool_uses = (
                    np.random.zipf(a=k, size=self.num_patterns) - 1
                )  # sampled index start from 1, but previous result is still valid
                valid_tool_uses = [t for t in tool_uses if t < self.num_patterns]
                hist.extend(valid_tool_uses[: self.total_num_requests - len(hist)])

            # Normal distribution
            # x = np.arange(0, self.num_patterns)
            # xU, xL = x + 0.5, x - 0.5
            # prob = ss.norm.cdf(xU, scale = 3, loc=self.num_patterns//2) - ss.norm.cdf(xL, scale = 3, loc=self.num_patterns//2)
            # prob = prob / prob.sum() # normalize the probabilities so their sum is 1
            # hist = np.random.choice(x, size = self.total_num_requests, p = prob)
            import matplotlib.pyplot as plt

            plt.hist(hist, bins=self.num_patterns)
            plt.savefig(f"zipf_distribution_{k}.png")
            tool_usage = defaultdict(int)
            for tool_index in hist:
                tool_usage[tool_index] += 1
            workload = []
            for tool_index, num_requests in tool_usage.items():
                prefix = prefix_stats[tool_index][0]
                selected_instances = np.random.choice(
                    self.data[prefix], num_requests, replace=True
                )
                for e in selected_instances:
                    output_len = len(self.tokenizer(e["output"]).input_ids)
                    workload.append(
                        {
                            "text": e["prompt"],
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": output_len,
                            },
                        }
                    )
        elif self.load_dist == LoadDistribution.NORMAL:
            assert k is not None
            prefix_stats = sorted(
                [(p, len(l)) for p, l in self.data.items()],
                key=lambda x: x[1],
                reverse=True,
            )[: self.num_patterns]
            # Normal distribution
            x = np.arange(0, self.num_patterns)
            xU, xL = x + 0.5, x - 0.5
            prob = ss.norm.cdf(xU, scale=k, loc=self.num_patterns // 2) - ss.norm.cdf(
                xL, scale=k, loc=self.num_patterns // 2
            )
            prob = prob / prob.sum()
            hist = np.random.choice(x, size=self.total_num_requests, p=prob)
            import matplotlib.pyplot as plt

            plt.hist(hist, bins=self.num_patterns)
            plt.savefig(f"normal_distribution_{k}.png")
            tool_usage = defaultdict(int)
            for tool_index in hist:
                tool_usage[tool_index] += 1
            workload = []
            for tool_index, num_requests in tool_usage.items():
                prefix = prefix_stats[tool_index][0]
                selected_instances = np.random.choice(
                    self.data[prefix], num_requests, replace=True
                )
                for e in selected_instances:
                    output_len = len(self.tokenizer(e["output"]).input_ids)
                    workload.append(
                        {
                            "text": e["prompt"],
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": output_len,
                            },
                        }
                    )
        else:
            raise NotImplementedError()
        self.add_input_token_ids_to_workload(workload)
        random.shuffle(workload)
        return workload


@dataclass
class Oracle(CustomRuntimeSelector):
    num_workloads: int
    trace = {}

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None):
        num_nodes = self.num_nodes
        self.trace[request_id] = text[:50]
        for i in range(self.num_workloads):
            if text.startswith(f"Workload {i} "):
                return i % num_nodes

        return random.randint(0, num_nodes - 1)

@dataclass
class OracleHotCold(CustomRuntimeSelector):
    num_workloads: int
    trace = {}

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None):
        num_nodes = self.num_nodes
        if num_nodes == 1:
            return 0
        self.trace[request_id] = text[:50]
        for i in range(self.num_workloads):
            if text.startswith(f"Workload {i} "):
                # return i % (num_nodes // 2)
                # return i % 2
                # return i % (num_nodes - 1)
                # return num_nodes - 1
                return 0

        # return num_nodes - 1 
        return random.randint(1, 3)
        # return random.randint(num_nodes // 2, num_nodes - 1)

@dataclass
class TBOracle:
    trace = {}
    tbl = {}
    num_nodes: int
    counter = {}

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None):
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

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None):
        match = re.search(r"You have access of the following tools:\n1.(.+?): ", text)
        if match:
            tool = match.group(1)
            if tool not in self.tbl:
                self.tbl[tool] = self.counter % self.num_nodes
                self.counter += 1
            return self.tbl[tool]
        else:
            return random.randint(0, self.num_nodes - 1)


class LooGLEDatasetType(Enum):
    LONG_QA = auto()
    SHORT_CLOZE = auto()
    SHORT_QA = auto()


class LooGLEDataset(DataLoader):
    def __init__(
        self,
        loogle_dataset_type: LooGLEDatasetType,
        num_patterns: int,
        total_num_requests: int,
        tokenizer,
        crop_max_decode=True,
    ):
        super().__init__(
            "loogle",
            num_patterns,
            total_num_requests,
            tokenizer=tokenizer,
        )
        self.prompt_format = {
            LooGLEDatasetType.SHORT_QA: "Please answer the question based on the long texts below. \n{input}\nQuestion: {Q}\nAnswer: ",
            LooGLEDatasetType.LONG_QA: "Please answer the question based on the long texts below. \n{input}\nQuestion: {Q}\nAnswer: ",
        }
        self.loogle_dataset_type = loogle_dataset_type
        self.data = self.read_data(loogle_dataset_type)
        self.max_decode_loogle = {  # based on filtring 1.5x IQR on dataset
            LooGLEDatasetType.SHORT_QA: 35,
            LooGLEDatasetType.LONG_QA: 28,
        }
        if not crop_max_decode:
            self.max_decode_loogle = {  # based on filtring 1.5x IQR on dataset
                LooGLEDatasetType.SHORT_QA: float("inf"),
                LooGLEDatasetType.LONG_QA: float("inf"),
            }
        # Short QA has about

    def read_data(
        self, LooGLE_dataset_type: LooGLEDatasetType = LooGLEDatasetType.SHORT_QA
    ):
        if LooGLE_dataset_type == LooGLEDatasetType.LONG_QA:
            data = load_dataset("bigainlco/LooGLE", "longdep_qa", split="test")
        elif LooGLE_dataset_type == LooGLEDatasetType.SHORT_QA:
            data = load_dataset("bigainlco/LooGLE", "shortdep_qa", split="test")
        self.data = data
        self.prompt_format = self.prompt_format[LooGLE_dataset_type]
        return data

    def generate_workload(self, max_length: int):
        workload = []
        max_num_patterns = len(self.data)
        logging.debug(f"Total patterns available {max_num_patterns}")
        if self.num_patterns > max_num_patterns:
            logging.warning(
                f"num_patterns {self.num_patterns} is larger than the number of patterns in the dataset {max_num_patterns}."
            )
            self.num_patterns = max_num_patterns

        for item in self.data.shuffle().select(range(self.num_patterns)):
            raw_inputs = item["input"]
            for j in eval(item["qa_pairs"]):
                json_obj = {"Q": j["Q"], "input": raw_inputs}
                prompt = self.prompt_format.format(**json_obj)
                # tokenized_prompt = self.tokenizer.encode(prompt)
                # if len(tokenized_prompt) > max_length:
                #     half = int(max_length/2)
                #     prompt = self.tokenizer.decode(tokenized_prompt[:half])+ self.tokenizer.decode(tokenized_prompt[-half:])
                #     tokenized_prompt = self.tokenizer.encode(prompt)
                workload.append(
                    {
                        "text": prompt,
                        "output": j["A"],
                        "sampling_params": {
                            "temperature": 0,
                        },
                    }
                )

        def tokenize_workload(request):
            input_ids = self.tokenizer(request["text"]).input_ids
            max_new_tokens = len(self.tokenizer(request["output"]).input_ids)
            request.pop("output")
            if max_new_tokens > self.max_decode_loogle[self.loogle_dataset_type]:
                max_new_tokens = self.max_decode_loogle[self.loogle_dataset_type]
            request["sampling_params"]["max_new_tokens"] = max_new_tokens
            request["input_ids"] = input_ids
            if len(input_ids) > max_length:
                half = int(max_length / 2)
                new_prompt = self.tokenizer.decode(
                    input_ids[:half]
                ) + self.tokenizer.decode(input_ids[-half:])
                new_tokenized_prompt = self.tokenizer.encode(new_prompt)
                request["text"] = new_prompt
                request["input_ids"] = new_tokenized_prompt
            return request

        with ThreadPoolExecutor(64) as executor:
            futures = []
            for request in workload:
                futures.append(executor.submit(tokenize_workload, request))
            wait(futures)
        random.shuffle(workload)
        return workload


@dataclass
class LoogleOracle(CustomRuntimeSelector):
    def __post_init__(self):
        self.trace = {}
        self.tbl = {}
        self.counter = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None):
        match = re.search(r"(.*)Question:", text, re.DOTALL)
        if match:
            tool = match.group(1)
            if tool not in self.tbl:
                self.tbl[tool] = self.counter % self.num_nodes
                self.counter += 1
            return self.tbl[tool]
        else:
            return random.randint(0, self.num_nodes - 1)
