from multi_node_loader import MultiNodeLoader
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy

# %%
mulit_node_loader = MultiNodeLoader(available_cuda_nodes=[0, 1])
model1 = "mistralai/Mistral-7B-v0.1"
model_details = mulit_node_loader.load_model(
    model1, gpus=[0, 1], urls=[]
)
model_details.update_runtime_selection_policy(DataParallelRuntimeSelectionPolicy.RANDOM)
res1 = model_details.generate_request(text="What is the capital of France", sampling_params={"max_new_tokens": 256})
res2 = model_details.generate_request(text="What is the capital of France", sampling_params={"max_new_tokens": 256})

print(res1, res2)
# Run Async Tests
# %%
mulit_node_loader.unload_model(model_details)


# Experiments
# Try many prefixes
# Try larger model weights
