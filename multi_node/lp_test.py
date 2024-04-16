# %%
from greedy_lp import LPGurobiGreedyTraversal, LPRadixCache
from sglang.srt.managers.router.model_runner import GPUConfig
from benchmarks.exp_configs.react_simulator_config import add_simulation_to_gpu_config
from transformers import AutoTokenizer

# %%
num_workloads = 100
num_requests = 4096
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
gpu_configs = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False),
    GPUConfig(gpu_id=0, url=None, use_ssh=False),
]
add_simulation_to_gpu_config(gpu_configs)
lorem = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id
est laborum."""

# %%
data_size = 30
lorem5 = " ".join([lorem for _ in range(data_size)])
texts = [
    f"Workload 1. {lorem5} A B C D",
    f"Workload 2. {lorem5} A B C D", 
    f"Workload 1. {lorem5} A B C D example 1",
    f"Workload 1. {lorem5} A B C D example 2",
    f"Workload 1. {lorem5} A B C D example 2 example 3",
    f"Workload 2. {lorem5} A B C D E",
    f"Workload 2. {lorem5} A B C D E F",
    f"Workload 2. {lorem5} A B C D E F G",
]
input_ids = [tokenizer.encode(text) for text in texts]
global_cache = LPRadixCache()
lp_tree_traversal = LPGurobiGreedyTraversal(2, gpu_configs=gpu_configs)
runtime_selected_per_workload = {
    "Workload 1": [],
    "Workload 2": []
}
for i in range(len(texts)):
    node = lp_tree_traversal.insert_into_cache_and_solve(
        tree_cache=global_cache, 
        input_ids=tuple(input_ids[i]), 
        decode_cost=0
    )
    gpu_selections = node.gpu_selections
    lp_tree_traversal.pretty_print(global_cache.root_node, tokenizer=tokenizer)
    print(f"GPU Selections: {gpu_selections}")
    if "Workload 1" in texts[i]:
        runtime_selected_per_workload["Workload 1"].append(list(gpu_selections)[0])
    else:
        runtime_selected_per_workload["Workload 2"].append(list(gpu_selections)[0])

# %%



