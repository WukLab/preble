from transformers import AutoTokenizer
from benchmarks.benchmark_workload_gen import get_react_workload

# model_name = "lmsys/vicuna-13b-v1.5"
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_workloads = 10
prompt = get_react_workload(f'Workload {num_workloads} ')
print(len(tokenizer.encode(prompt)))