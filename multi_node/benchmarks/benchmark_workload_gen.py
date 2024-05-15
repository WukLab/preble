import random
import json
import string
import uuid
import pandas as pd
import math


import numpy as np
import random
from tqdm import tqdm
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
from benchmarks import chameleon
from benchmarks import toolqa
import os
import pandas as pd

logger = logging.getLogger(__name__)

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
    
    def get_text(self, request, tokenizer):
        text = tokenizer.decode(request["input_ids"])
        request["text"] = text

    def add_input_token_ids_to_workload(self, workload):
        with ThreadPoolExecutor(64) as executor:
            futures = []
            for request in workload:
                futures.append(executor.submit(self.get_token_ids, request, self.tokenizer))
            for future in futures:
                future.result()
    
    def add_text_from_token_ids_to_workload(self, workload):
        with ThreadPoolExecutor(64) as executor:
            futures = []
            results = list(tqdm(executor.map(self.get_text, workload, [self.tokenizer]*len(workload)), total=len(workload)))
            # for request in workload:
            #     futures.append(executor.submit(self.get_text, request, self.tokenizer))
            # for future in futures:
            #     future.result()
                
    def workload_specific_args(self):
        return {
            "num_patterns": self.num_patterns,
            "total_num_requests": self.total_num_requests,
            "load_dist": str(self.load_dist),
        }

    def get_tokenizer(self):
        return self.tokenizer
    
class PercentCommonSharedDataLoader(DataLoader):
    def __init__(
        self,
        num_patterns: int,
        total_num_requests: int,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        load_dist: LoadDistribution = LoadDistribution.EVEN,
        distribution_of_non_shared: float = 0.0,
        percent_of_common_shared: float = 1.0,
        output_len: int = 1,
        context_len: int = None, # if this is not none, ignore num_in_context_exampels
    ):
        super().__init__(
            "random", num_patterns, total_num_requests, tokenizer, load_dist
        )
        self.distribution_of_non_shared = distribution_of_non_shared
        self.output_len = output_len
        self.context_len = context_len
        self.percent_of_common_shared = percent_of_common_shared
    
    def generate_workload(self):
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
        
        common_prompt_lengths = math.floor(self.context_len * self.percent_of_common_shared)
        unique_sharing_length = self.context_len - common_prompt_lengths
        for i in range(num_prefixed_shared):
            workload_num = 1 + i % self.num_patterns
            prompt_tokens = [0] * common_prompt_lengths + [workload_num] * unique_sharing_length
            workload.append(
                {
                    "input_ids" : prompt_tokens,
                    "sampling_params": copy.deepcopy(sampling_params),
                    "rid": uuid.uuid4().hex,
                }
            )
        for i in range(num_non_shared):
            prompt_tokens = [1 + i + self.num_patterns] * self.context_len
            workload.append(
                {
                    "input_ids" : prompt_tokens,
                    "sampling_params": copy.deepcopy(sampling_params),
                    "rid": uuid.uuid4().hex,
                }
            )
        logger.info('Start decoding to text')
        self.add_text_from_token_ids_to_workload(workload)
        random.shuffle(workload)
        return workload
        
        

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
        random_workload_path=None,
        workload_start_from: int = 0,
        decoding_size=None,
        context_len: int = None, # if this is not none, ignore num_in_context_exampels
    ):
        super().__init__(
            "random", num_patterns, total_num_requests, tokenizer, load_dist
        )
        self.distribution_of_non_shared = distribution_of_non_shared
        self.output_len = output_len
        self.num_in_context_examples = num_in_context_examples
        self.random_workload_path=random_workload_path
        self.workload_start_from = workload_start_from
        self.decoding_size = decoding_size
        self.context_len = context_len
        if self.context_len:
            self.num_in_context_examples = math.ceil(self.context_len / 475)

    def generate_workload(self, k):
        num_prefixed_shared = int(
            self.total_num_requests * (1 - self.distribution_of_non_shared)
        )
        num_non_shared = int(self.total_num_requests * self.distribution_of_non_shared)
        workload = []
        output_len = self.output_len
        if self.decoding_size:
            output_len = self.decoding_size
        sampling_params = {
            "experiment_id": f"random_experiment_{self.num_patterns}_{self.distribution_of_non_shared}_{self.total_num_requests}",
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True, # For better micro-benchmark
        }
        for i in range(num_prefixed_shared):
            workload_num = self.workload_start_from + i % self.num_patterns
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
        # prompt_lens = [len(p["input_ids"]) for p in workload]
        # plt.hist(prompt_lens)
        # plt.savefig(f"react_prompt_length.png")
        return workload
    
    @staticmethod
    def is_hot(output):
        return output.prompt_text.startswith("Workload ")
    
    @staticmethod
    def get_prefix_index(output):
        match = re.search(r'\bWorkload\s+(\d+)', output.prompt_text)
        if match:
            return int(match.group(1))
        else:
            return None

    def workload_specific_args(self):
        return {
            "num_patterns": self.num_patterns,
            "total_num_requests": self.total_num_requests,
            "load_dist": str(self.load_dist),
            "random_ratio": self.distribution_of_non_shared,
            "output_len": self.output_len if not self.decoding_size else self.decoding_size,
            "num_in_context_examples": self.num_in_context_examples,
        }

class WorkloadPrefixShareGPTDataLoader(DataLoader):
    def __init__(
        self,
        num_patterns: int,
        total_num_requests: int,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        load_dist: LoadDistribution = LoadDistribution.EVEN,
        distribution_of_non_shared: float = 0.0,
        output_len: int = 1,
        num_in_context_examples: int = 4,
        random_workload_path=None,
        workload_start_from: int = 0,
    ):
        super().__init__(
            "random", num_patterns, total_num_requests, tokenizer, load_dist
        )
        self.distribution_of_non_shared = distribution_of_non_shared
        self.output_len = output_len
        self.num_in_context_examples = num_in_context_examples
        self.random_workload_path=random_workload_path
        self.workload_start_from = workload_start_from

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
            workload_num = self.workload_start_from + i % self.num_patterns
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

        random_workload = generate_random_workload(random_workload_path=self.random_workload_path)
        for _ in range(num_non_shared):
            prompt = random.choice(random_workload)
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
    
    @staticmethod
    def get_prefix_index(output):
        match = re.search(r'\bWorkload\s+(\d+)', output.prompt_text)
        if match:
            return int(match.group(1))
        else:
            return None

    def workload_specific_args(self):
        return {
            "num_patterns": self.num_patterns,
            "total_num_requests": self.total_num_requests,
            "load_dist": str(self.load_dist),
            "random_ratio": self.distribution_of_non_shared,
            "output_len": self.output_len,
            "num_in_context_examples": self.num_in_context_examples,
        }

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
            if len(prefix_stats) < self.num_patterns:
                logger.info(f'Asking for too many prefixes with large sharing')
                selected_prefixs = np.random.choice(
                    prefix_stats, self.num_patterns, replace=True
                )
            else:
                selected_prefixs = np.random.choice(
                    prefix_stats, self.num_patterns, replace=False
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
        elif self.load_dist == LoadDistribution.ALL:
            load_threshold = math.ceil(self.total_num_requests // self.num_patterns)
            prefix_stats = [p for p, l in self.data.items() if len(l) >= load_threshold]
            selected_prefixs = np.random.choice(
                prefix_stats, self.num_patterns, replace=True
            )
            for p in selected_prefixs:
                selected_instances = self.data[p]
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

    def workload_specific_args(self):
        return {
            "num_patterns": self.num_patterns,
            "total_num_requests": self.total_num_requests,
            "load_dist": str(self.load_dist),
        }

class VideoOracle(CustomRuntimeSelector):
    trace = {}
    counter = {}
    rr = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None, *args, **kwargs):
        video = input_ids[30]
        self.counter[video] = self.counter.get(video, 0) + 1
        num_nodes = self.num_nodes
        return video % num_nodes
        
class VideoDataLoader(DataLoader):
    def __init__(
        self,
        data_path: str,
        total_num_requests: int,
        max_shared_prompt_token_length: int,
        num_patterns: int,
        tokenizer,
    ):
        super().__init__(data_path = data_path,
                       num_patterns = 0, 
                       total_num_requests = total_num_requests,
                       tokenizer = tokenizer,
                       load_dist = None)
        
        self.sysprompt = "Please answer the questions based on the following information before the question."
        self.max_shared_prompt_token_length = max_shared_prompt_token_length
        self.num_patterns = num_patterns
        self.sysprompt_tokens = self.tokenizer.encode(self.sysprompt)
        print(len(self.sysprompt_tokens))
        self.data = self.read_data()
        return
    
    def read_data(
        self
    ):
        df = pd.read_csv(self.data_path)
        frame_counts = df["frame_count"]
        vids = df["video"]
        questions = df["question"]
        selected_answers = df["answer"]
        video_token = 0
        vid_to_token = {}
        data = []
        for i in range(len(frame_counts)):
            frame_count = frame_counts[i]
            num_tokens = int(frame_count / 30 * 256)
            options = [df["a" + str(k)][i] for k in range(5)]
            question = questions[i] + " " + " ".join(options)
            answer = df["a" + str(selected_answers[i])][i]
            vid = vids[i]
            if vid in vid_to_token:
                token = vid_to_token[vid]
            else:
                vid_to_token[vid] = video_token
                token = video_token
                video_token += 1
            cropped_video_len = min(self.max_shared_prompt_token_length - len(self.sysprompt_tokens), num_tokens)
            prompt_tokens = self.sysprompt_tokens + [token] * cropped_video_len
            data.append([prompt_tokens, question, answer])
        if len(data) < self.num_patterns:
            logger.info("Asking for too many prefixes patterns")
        return data
    
    def generate_workload(self):
        workload = []
        selected_video_qa = np.random.choice(
            np.arange(len(self.data)), self.num_patterns, replace=False
        )
        total_prompt_length = sum(len(self.data[idx][0]) for idx in selected_video_qa)
        qa_per_video = [math.ceil(self.total_num_requests / total_prompt_length * len(self.data[idx][0])) for idx in selected_video_qa] 
        print('STD of example per prefix', np.std(qa_per_video))
        cnt = 0
        total_prefix_length = 0
        for i, idx in enumerate(selected_video_qa):
            prompt_tokens, question, answer = self.data[idx]
            max_new_tokens = len(self.tokenizer(answer).input_ids)
            print(len(prompt_tokens), max_new_tokens, answer, qa_per_video[i])
            total_prefix_length += len(prompt_tokens)
            for _ in range(qa_per_video[i]):
                question_tokens = self.tokenizer.encode(uuid.uuid4().hex + " " + question)
                workload.append(
                    {
                        # "text": self.tokenizer.decode(prompt_tokens + question_tokens),
                        "input_ids" : prompt_tokens + question_tokens,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens" : max_new_tokens
                        },
                    }
                )
                cnt += 1
            # request["sampling_params"]["max_new_tokens"] = max_new_tokens
        
        logger.info(f'total prefix: {total_prefix_length}, start decoding for prompt text')
        self.add_text_from_token_ids_to_workload(workload)
        return workload

@dataclass
class Oracle(CustomRuntimeSelector):
    num_workloads: int
    trace = {}
    rr = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None, *args, **kwargs):
        num_nodes = self.num_nodes
        self.trace[request_id] = text[:50]
        for i in range(self.num_workloads):
            if text.startswith(f"Workload {i} "):
                return i % num_nodes
        self.rr = (self.rr + 1) % num_nodes
        return self.rr

@dataclass
class OracleHotCold(CustomRuntimeSelector):
    num_workloads: int
    trace = {}
    cold_cnt = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None, *args, **kwargs):
        num_nodes = self.num_nodes
        if num_nodes == 1:
            return 0
        self.trace[request_id] = text[:50]
        for i in range(self.num_workloads):
            if text.startswith(f"Workload {i} "):
                # return i % (num_nodes // 2)
                # return i % 3
                # return i % (num_nodes - 1)
                # return num_nodes - 1
                return 0
        # self.cold_cnt = (self.cold_cnt + 1) % 11
        # if self.cold_cnt < 2:
        #     return 0
        # if self.cold_cnt < 4:
        #     return 1
        # if self.cold_cnt < 6:
        #     return 2
        # else:
        #     return 3
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

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None, *args, **kargs):
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
        max_tokens_override=None,
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
        self.max_tokens_override = max_tokens_override
        # Short QA has about

    def read_data(
        self, LooGLE_dataset_type: LooGLEDatasetType = LooGLEDatasetType.SHORT_QA
    ):
        if LooGLE_dataset_type == LooGLEDatasetType.LONG_QA:
            data = load_dataset("bigainlco/LooGLE", "longdep_qa", split="test")
        elif LooGLE_dataset_type == LooGLEDatasetType.SHORT_QA:
            # data = load_dataset("bigainlco/LooGLE", "shortdep_qa", split="test", download_mode='force_redownload')
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
        print(f"Generating workload for {self.num_patterns} patterns")
        
        sampled_dataset = self.data.shuffle().select(range(self.num_patterns))
        qa_pairs = [eval(item['qa_pairs']) for item in sampled_dataset]
        num_raw_requests = sum(len(qas) for qas in qa_pairs)
        # print(qa_pairs, num_raw_requests)
        
        #NOTE: loogle dataset has not enought QAs for each document
        #      We replicate QAs w.r.t existing prefix sharing distributions
        scale_factor = self.total_num_requests / num_raw_requests
        request_pre_prefix = [0] * len(sampled_dataset)
        for i, item in tqdm(enumerate(sampled_dataset)):
            raw_inputs = item["input"]
            # if i == 0:
            #     print(raw_inputs)
            num_qa_pairs = len(qa_pairs[i])
            for k in range(math.ceil(num_qa_pairs * scale_factor)):
                request_pre_prefix[i] += 1
                j = qa_pairs[i][k % num_qa_pairs]
                json_obj = {"Q": uuid.uuid4().hex + j["Q"], "input": raw_inputs}
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
        print(f"STD loogle requests per prefix: {np.std(request_pre_prefix)}")

        def tokenize_workload(request):
            input_ids = self.tokenizer(request["text"]).input_ids
            max_new_tokens = len(self.tokenizer(request["output"]).input_ids)
            request.pop("output")
            if max_new_tokens > self.max_decode_loogle[self.loogle_dataset_type]:
                max_new_tokens = self.max_decode_loogle[self.loogle_dataset_type]
            request["sampling_params"]["max_new_tokens"] = max_new_tokens
            if self.max_tokens_override:
                request["sampling_params"]["max_new_tokens"] = self.max_tokens_override
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

            # Wrap tqdm around the futures to display a progress bar
            for future in tqdm(futures, desc="Processing workload"):
                # Using as_completed would be more typical for tqdm, but here we're just ensuring completion
                future.result()


        random.shuffle(workload)
        return workload

    def workload_specific_args(self):
        return {
            "num_patterns": self.num_patterns,
            "total_num_requests": self.total_num_requests,
        }

@dataclass
class LoogleOracle(CustomRuntimeSelector):
    def __post_init__(self):
        self.trace = {}
        self.tbl = {}
        self.counter = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None, *args, **kwargs):
        match = re.search(r"(.*)Question:", text, re.DOTALL)
        if match:
            tool = match.group(1)
            if tool not in self.tbl:
                self.tbl[tool] = self.counter % self.num_nodes
                self.counter += 1
            return self.tbl[tool]
        else:
            return random.randint(0, self.num_nodes - 1)
        
@dataclass
class ProgrammingOracle(CustomRuntimeSelector):
    def __post_init__(self):
        self.trace = {}
        self.tbl = {}
        self.counter = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None, *args, **kwargs):
        if text not in self.tbl:
            self.tbl[text] = self.counter % self.num_nodes
            self.counter += 1
        return self.tbl[text]


class MultiDomainToolBenchDataLoader(DataLoader):
    def __init__(
        self,
        data_path: str,
        num_patterns: int,
        total_num_requests: int,
        num_domains: int,
        domain_size: int,
        tokenizer,
        load_dist: LoadDistribution = LoadDistribution.EVEN,
    ):
        super().__init__(
            data_path, num_patterns, total_num_requests, tokenizer, load_dist
        )
        self.data = self.read_data()
        self.num_domains = num_domains
        self.domain_size = domain_size

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
        elif self.load_dist == LoadDistribution.ALL:
            load_threshold = math.ceil(self.total_num_requests // self.num_patterns)
            prefix_stats = [p for p, l in self.data.items() if len(l) >= load_threshold]
            selected_prefixs = np.random.choice(
                prefix_stats, self.num_patterns, replace=True
            )
            for p in selected_prefixs:
                selected_instances = self.data[p]
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
        new_workload_with_domains = []
        domain_string = []
        for i in range(self.num_domains):
            random_domain_string = ""
            if self.domain_size != 0:
                random_domain_string = "ID: " + gen_random_string(self.domain_size) + " "
            domain_string.append(f"Domain: {i} {random_domain_string}")
        for domain_num in range(self.num_domains):
            for item in workload:
                new_text_prompt = domain_string[domain_num] + item["text"]
                new_workload_with_domains.append({
                    "text": new_text_prompt,
                    "sampling_params": item["sampling_params"]
                })
        self.add_input_token_ids_to_workload(new_workload_with_domains)
        random.shuffle(new_workload_with_domains)
        return new_workload_with_domains

@dataclass
class TBMultiDomainOracle(CustomRuntimeSelector):
    trace = {}
    tbl = {}
    counter: int = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None):
        match = re.search(r"Domain: (.+?) ", text)
        if match:
            tool = match.group(1)
            if tool not in self.tbl:
                self.tbl[tool] = self.counter % self.num_nodes
                self.counter += 1
            return self.tbl[tool]
        else:
            return random.randint(0, self.num_nodes - 1)
        

class ChameleonTabMWPLoader(DataLoader):
    """DataLoader for Chameleon + TabMWP dataset"""

    def __init__(self, data_path: str, num_patterns: int,
                 tokenizer: PreTrainedTokenizer):
        super().__init__(data_path, num_patterns, None, tokenizer)
        self.data = self.read_data(data_path)
        self.pattern_req_groups = self.load_data_by_pattern()

    def read_data(self, data_path: str):
        data = []
        with open(data_path, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    def generate_module_prediction_request(self, sample):
        return {
            'text': chameleon.prompt_policy.prompt.strip() + '\n\n' + sample['modules:input'],
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': len(self.tokenizer(str(sample['modules:output'])).input_ids),
            },
        }

    def generate_row_lookup_request(self, sample):
        return {
            'text': chameleon.prompt_rl.prompt.strip() + '\n\n' + sample['row_lookup:input'],
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': len(self.tokenizer(sample['row_lookup:output']).input_ids),
            },
        }

    def generate_column_lookup_request(self, sample):
        return {
            'text': chameleon.prompt_cl.prompt.strip() + '\n\n' + sample['column_lookup:input'],
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': len(self.tokenizer(sample['column_lookup:output']).input_ids),
            },
        }
    
    def generate_table_verbalizer_request(self, sample):
        return {
            'text': chameleon.prompt_tv.prompt.strip() + '\n\n' + sample['table_verbalizer:input'],
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': len(self.tokenizer(sample['table_verbalizer:output']).input_ids),
            },
        }
    
    def generate_knowledge_retrieval_request(self, sample):
        return {
            'text': chameleon.prompt_kr.prompt.strip() + '\n\n' + sample['knowledge_retrieval:input'],
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': len(self.tokenizer(sample['knowledge_retrieval:output']).input_ids),
            },
        }

    def generate_program_generator_request(self, sample):
        if sample['example']['choices']:
            demo_prompt = chameleon.prompt_pg.prompt_choice.strip()
        else:
            demo_prompt = chameleon.prompt_pg.prompt_free.strip()
        return {
            'text': demo_prompt + '\n\n' + sample['program_generator:input'],
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': len(self.tokenizer(sample['program_generator:output']).input_ids),
            },
        }

    def generate_program_generator_verifier_request(self, sample):
        if sample['example']['choices']:
            demo_prompt = chameleon.prompt_pg.prompt_choice.strip()
        else:
            demo_prompt = chameleon.prompt_pg.prompt_free.strip()
        return {
            'text': demo_prompt + '\n\n' + sample['program_generator_and_verifier:input'],
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': len(self.tokenizer(sample['program_generator_and_verifier:output']).input_ids),
            },
        }
    
    def generate_solution_generator_request(self, sample):
        if sample['example']['choices']:
            demo_prompt = chameleon.prompt_sg.prompt_choice.strip()
        else:
            demo_prompt = chameleon.prompt_sg.prompt_free.strip()
        return {
            'text': demo_prompt + '\n\n' + sample['solution_generator:input'],
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': len(self.tokenizer(f"The answer is {sample['solution_generator:output']}.").input_ids),
            },
        }

    def load_data_by_pattern(self):
        pattern_req_groups = {}
        for i, sample in enumerate(self.data):
            # handle predict module requests
            pattern_req_groups['module_prediction'] = pattern_req_groups.get('module_prediction', []) \
                + [self.generate_module_prediction_request(sample)]
            
            modules = sample['modules:output']
            if "row_lookup" in modules and sample['row_lookup:input']:
                pattern_req_groups['row_lookup'] = pattern_req_groups.get('row_lookup', []) \
                    + [self.generate_row_lookup_request(sample)]
            if "column_lookup" in modules and sample['column_lookup:input']:
                pattern_req_groups['column_lookup'] = pattern_req_groups.get('column_lookup', []) \
                    + [self.generate_column_lookup_request(sample)]
            if "table_verbalizer" in modules:
                pattern_req_groups['table_verbalizer'] = pattern_req_groups.get('table_verbalizer', []) \
                    + [self.generate_table_verbalizer_request(sample)]
            if "knowledge_retrieval" in modules:
                pattern_req_groups['knowledge_retrieval'] = pattern_req_groups.get('knowledge_retrieval', []) \
                    + [self.generate_knowledge_retrieval_request(sample)]
            # program_generator and program_generator_and_verifier shares the same prompt
            if "program_generator" in modules:
                pattern_req_groups['program_generator_verifier'] = pattern_req_groups.get('program_generator_verifier', []) \
                    + [self.generate_program_generator_request(sample)]
            if "program_generator_and_verifier" in modules:
                pattern_req_groups['program_generator_verifier'] = pattern_req_groups.get('program_generator_verifier', []) \
                    + [self.generate_program_generator_verifier_request(sample)]
            if "solution_generator" in modules:
                pattern_req_groups['solution_generator'] = pattern_req_groups.get('solution_generator', []) \
                    + [self.generate_solution_generator_request(sample)]

        return pattern_req_groups
    
    def generate_workload(self, k: int = None):
        num_patterns = self.num_patterns
        if self.num_patterns > len(self.pattern_req_groups):
            print(f'Not enough patterns in the dataset. Only {len(self.pattern_req_groups)} patterns available.')
            num_patterns = len(self.pattern_req_groups)
        pattern_groups = random.sample(list(self.pattern_req_groups.keys()), num_patterns)
        requests = []
        for pattern in pattern_groups:
            requests += self.pattern_req_groups[pattern]
        random.shuffle(requests)
        requests = requests[:k]
        # decoding_lengths = []
        for request in requests:
            request['sampling_params']['max_new_tokens'] = 26
            # decoding_lengths.append(request['sampling_params']['max_new_tokens'])
        # Get stats of decoding lengths
        # print(f"Decoding mean: {np.mean(decoding_lengths)}, std: {np.std(decoding_lengths)} max: {np.max(decoding_lengths)}")
        self.add_input_token_ids_to_workload(requests)
        # input_lengths = []
        # for request in requests:
        #     input_lengths.append(len(request['input_ids']))

        # print(f"Input mean: {np.mean(input_lengths)}, std: {np.std(input_lengths)} max: {np.max(input_lengths)}")
        # breakpoint()

        if len(requests) < k:
            print(f'Not enough requests in the dataset. Only {len(requests)} requests available.')
        return requests
    

class CreatorMATHLoader(DataLoader):
    """DataLoader for Creator + MATH dataset.
    Since the agents depend on the previous steps, 
    we use the same tool and error message for all samples."""

    tool = """```python
from sympy import symbols, solve

def solve_equations():
    \"\"\"
    Solves the system of equations 3p + 4q = 8 and 4p + 3q = 13 using sympy.
    Returns: The value of q that satisfies both equations.
    \"\"\"
    p, q = symbols('p q')
    eq1 = 3*p + 4*q - 8
    eq2 = 4*p + 3*q - 13
    solution = solve((eq1, eq2), (p, q))
    return solution[q]

# Call the function to solve the equations
q = solve_equations()

# Print the answer
print("Final Answer:", q)"""

    error = """Traceback (most recent call last):
  File "code_exec/tmp0.py", line 15, in <module>
    z_over_y = solve_equations()
  File "code_exec/tmp0.py", line 12, in solve_equations
    return solution[z] / solution[y]
KeyError: z"""

    def __init__(self, data_path: str, 
                 tokenizer: PreTrainedTokenizer):
        super().__init__(data_path, None, None, tokenizer)
        self.data = self.read_data(os.path.join(data_path, 'dataset'))
        with open(os.path.join(data_path, 'prompt_lib/prompt_CREATOR_creation.md'), 'r') as f:
            self.create_prompt = f.read()
        with open(os.path.join(data_path, 'prompt_lib/prompt_CREATOR_decision.md'), 'r') as f:
            self.decide_prompt = f.read()
        with open(os.path.join(data_path, 'prompt_lib/prompt_rectification.md'), 'r') as f:
            self.correct_prompt = f.read()

    def read_data(self, data_path: str):
        data = []
        for file in os.listdir(data_path):
            with open(os.path.join(data_path, file), "r") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        return data

    def generate_workload(self, k: int = None):
        requests = []
        random.shuffle(self.data)
        for i, sample in enumerate(self.data):
            if k is not None and len(requests) >= k:
                break
            requests.append({
                'text': self.create_prompt.replace('==qst==', sample['question']),
                'sampling_params': {
                    'temperature': 0.0,
                    'max_new_tokens': 30,
                },
            })
            requests.append({
                'text': self.decide_prompt.replace('==qst==', sample['question']).replace(
                    '==tool==', self.tool
                ),
                'sampling_params': {
                    'temperature': 0.0,
                    'max_new_tokens': 30,
                },
            })
            requests.append({
                'text': self.correct_prompt.replace('==qst==', sample['question']).replace(
                    '==ori==', self.tool
                ).replace('==err==', self.error),
                'sampling_params': {
                    'temperature': 0.0,
                    'max_new_tokens': 30,
                },
            })
        
        self.add_input_token_ids_to_workload(requests)
        return requests[:k]
    

class ToolQALoader(DataLoader):
    """DataLoader for ToolQA dataset.
    
    The ToolQA dataset consists of two level of difficulty: easy and hard.
    In this class, workload is generated randomly from both levels.
    
    The original ToolQA paper uses multi-step reasoning (with langchain).
    Since we are not using actual LLMs, we generate the request of first step."""

    scratchpad = """\nThought 1:\n"""

    def __init__(self, data_path: str, 
                 tokenizer: PreTrainedTokenizer, ):
        super().__init__(data_path, None, None, tokenizer)
        self.data = self.read_data(data_path)

    def read_data(self, data_path: str):
        data = []
        for hardness in ['easy', 'hard']:
            for file in os.listdir(os.path.join(data_path, hardness)):
                with open(os.path.join(data_path, hardness, file), "r") as f:
                    for line in f:
                        if line.strip():
                            example = json.loads(line)
                            example['hardness'] = hardness
                            data.append(example)
        return data

    def generate_workload(self, k: int = None):
        requests = []
        for i, sample in enumerate(self.data):
            if k is not None and i >= k:
                break
            examples = toolqa.TOOLQA_EASY8 if sample['hardness'] == 'easy' else toolqa.TOOLQA_HARD3
            requests.append({
                'text': toolqa.REACT_INSTRUCTION.format(examples=examples, 
                                                 question=sample['question'], 
                                                 scratchpad=self.scratchpad),
                'sampling_params': {
                    'temperature': 0.0,
                    'max_new_tokens': 30,
                },
            })
        
        self.add_input_token_ids_to_workload(requests)
        return requests


class VirtualEnvLoader(DataLoader):
    """DataLoader for VirtualEnv dataset."""

    def __init__(self, data_path: str, num_patterns: int,
                 tokenizer: PreTrainedTokenizer,total_num_requests=None):
        super().__init__(data_path, num_patterns, None, tokenizer)
        self.data = self.read_data(data_path)
        self.total_num_requests = total_num_requests
        self.tokenizer = tokenizer

    def read_data(self, data_path: str):
        with open(data_path, 'r') as f:
            return json.load(f)

    def generate_workload(self, k: int = None) -> List[List[dict]]:
        """Return:
        - a list of list of requests, where each list of requests is a conversation.
        """
        random.shuffle(self.data)
        k = k if k is not None else len(self.data)
        num_patterns = self.num_patterns
        if self.num_patterns > len(self.data):
            print(f'Not enough patterns in the dataset. Only {len(self.data)} patterns available.')
            num_patterns = len(self.data)
        num_raw_questions = sum(len(self.data[i % len(self.data)]) for i in range(self.num_patterns))
        scale_factor = self.total_num_requests / num_raw_questions
        
        sample_response = self.tokenizer.decode([1 for _ in range(14)])
        
        requests = []
        for i in range(self.num_patterns):
            env_id = f"Environment ID {i} "
            sample = self.data[i % len(self.data)]
            req_group = []
            last_turn = None
            for j in range(math.ceil(len(sample) * scale_factor)):
                # if j > 50:
                #     continue
                if j >= len(sample):
                    turn = {"prompt": last_turn["prompt"] + sample_response}
                else:
                    turn = sample[j]
                last_turn = turn
                req_group.append({
                    'text': env_id + turn['prompt'],
                    'sampling_params': {
                        'temperature': 0.0,
                        'max_new_tokens': 26,
                    },
                })
            requests.append(req_group)

        for req_group in requests:
            self.add_input_token_ids_to_workload(req_group)
        random.shuffle(requests)
        return requests

@dataclass
class VirtualenvOracle(CustomRuntimeSelector):
    num_workloads: int
    trace = {}
    rr = 0

    def runtime_selector(self, text: str, request_id: str, input_ids: List = None, sampling_params=None, *args, **kwargs):
        num_nodes = self.num_nodes
        self.trace[request_id] = text[:50]
        for i in range(self.num_workloads):
            if text.startswith(f"Environment ID {i} "):
                return i % num_nodes
        self.rr = (self.rr + 1) % num_nodes
        return self.rr

class ProgrammingDataset(DataLoader):
    def __init__(
        self,
        num_patterns: int,
        total_num_requests: int,
        tokenizer,
        crop_max_decode=True,
        max_tokens_override=None,
        shared_length=2400
    ):
        super().__init__(
            "programming",
            num_patterns,
            total_num_requests,
            tokenizer=tokenizer,
        )
        self.data = self.read_data()
        self.max_tokens_override = max_tokens_override
        self.p_90_sample_lengths_of_dataset = 12979
        self.shared_length = shared_length

        # Short QA has about

    def read_data(
        self
    ):
        ds = load_dataset("codeparrot/apps", split="test", difficulties=["introductory"])
        self.data = ds
        return self.data

    def generate_workload(self, max_length: int):
        workload = []
        max_num_patterns = len(self.data)
        logging.debug(f"Total patterns available {max_num_patterns}")
        if self.num_patterns > max_num_patterns:
            logging.warning(
                f"num_patterns {self.num_patterns} is larger than the number of patterns in the dataset {max_num_patterns}."
            )
            self.num_patterns = max_num_patterns
        print(f"Generating workload for {self.num_patterns} patterns")
        
        sampled_dataset = self.data

        num_raw_requests = 0
        new_sampled_dataset = []
        i = 0
        while len(new_sampled_dataset) != self.num_patterns:
            item = sampled_dataset[i]
            try:
                solution = json.loads(item["solutions"])
                num_raw_requests += len(solution)
                input_output = json.loads(item["input_output"]) # Test to hande errors
                new_sampled_dataset.append(item)
            except Exception as e:
                print(e)
                continue
            i += 1
        scale_factor = self.total_num_requests / num_raw_requests

        shared_prompt = self.tokenizer.decode([1 for _ in range(self.shared_length)])
        logging.info(f"Length of new sampled dataset actually has {len(new_sampled_dataset)} patterns")

        workload = []
        for i in tqdm(range(len(new_sampled_dataset))):
            sample = new_sampled_dataset[i]
            sample["solutions"] = json.loads(sample["solutions"])
            sample["input_output"] = json.loads(sample["input_output"])
            input_output_strings = ""
            for input in sample["input_output"]["inputs"]:
                input_output_strings += str(input) + "\n"
            for output in sample["input_output"]["outputs"]:
                input_output_strings += str(output) + "\n"

            if not sample.get("fn_name"):
                input_format = "\nUse Standard Input format"#\n"
            else:
                input_format = "\nUse Call-Based format"#\n"
            question = sample["question"]
            sample_prompt = f"Question: {question} {input_format} {input_output_strings} \n\nAnswer:\n"
            if len(sample_prompt) > self.p_90_sample_lengths_of_dataset: # p90 
                continue
            num_qa_pairs = len(sample["solutions"])
            for solution_i in range(math.ceil(num_qa_pairs * scale_factor)):
                solution = sample["solutions"][solution_i % num_qa_pairs]
                workload.append(
                    {
                        "text": shared_prompt + sample_prompt + uuid.uuid4().hex, # Make each request unique
                        "output": solution,
                        "sampling_params": {
                            "temperature": 0,
                        },
                    }
                )
        def tokenize_workload(request):
            input_ids = self.tokenizer(request["text"], max_length=32768, truncation=True).input_ids
            max_new_tokens = len(self.tokenizer(request["output"], max_length=32768, truncation=True).input_ids)
            request.pop("output")
            if max_new_tokens > self.max_tokens_override:
                max_new_tokens = self.max_tokens_override
            request["sampling_params"]["max_new_tokens"] = max_new_tokens
            # if self.max_tokens_override:
                # request["sampling_params"]["max_new_tokens"] = self.max_tokens_override
            request["input_ids"] = input_ids
            if len(input_ids) > 32768 - max_new_tokens:
                request["input_ids"] = input_ids[:32768 - max_new_tokens]
            return request
    
        with ThreadPoolExecutor(64) as executor:
            futures = []
            for request in workload:
                futures.append(executor.submit(tokenize_workload, request))

            # Wrap tqdm around the futures to display a progress bar
            for future in tqdm(futures, desc="Processing workload"):
                # Using as_completed would be more typical for tqdm, but here we're just ensuring completion
                future.result()


        random.shuffle(workload)
        return workload

    def workload_specific_args(self):
        return {
            "num_patterns": self.num_patterns,
            "total_num_requests": self.total_num_requests,
            "shared_length": self.shared_length
        }



def load_realistic_send_out_times(azure_llm_infernce_trace_dir="datasets/dataset_exploration"):
    TRACE_NAMES = [
        "Coding",
        "Conversation",
    ]
    TRACE_FILENAMES = [
        "AzureLLMInferenceTrace_code.csv",
        "AzureLLMInferenceTrace_conv.csv",
    ]
    # Read all traces
    df_traces = {}
    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(os.path.join(azure_llm_infernce_trace_dir, trace_filename), parse_dates=["TIMESTAMP"])
    convo_trace = df_traces["Conversation"]
    first_timestamp = convo_trace['TIMESTAMP'].iloc[0]
    convo_trace['TIMESTAMP'] = convo_trace['TIMESTAMP'] - first_timestamp
    convo_trace['TIMESTAMP'] = convo_trace['TIMESTAMP'].dt.total_seconds()
    convo_trace.drop("ContextTokens", axis=1, inplace=True)
    convo_trace.drop("GeneratedTokens", axis=1, inplace=True)
    convo_trace = convo_trace[convo_trace['TIMESTAMP'] != 0.0]
    first_timestamp = convo_trace['TIMESTAMP'].iloc[0]
    convo_trace['TIMESTAMP'] = convo_trace['TIMESTAMP'] - first_timestamp
    send_out_times = list(convo_trace["TIMESTAMP"])
    return send_out_times
