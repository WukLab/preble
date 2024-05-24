import sys
import os
import requests
import concurrent.futures
import math
from argparse import ArgumentParser
from typing import Iterable, List
import numpy as np
import asyncio
import time, datetime
import aiohttp
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../multi_node")))
from benchmarks.benchmark_workload_gen import get_react_workload
from multi_node_loader import MultiNodeLoader, ModelDetails

log = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

   
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model_name = "mistralai/Mistral-7B-v0.1"
    available_gpus = [0, 1]
    loader = MultiNodeLoader(available_cuda_nodes=available_gpus)
    model_details = loader.load_model(
        model_name, gpus=available_gpus, urls=[], freeze=True,
    )
    