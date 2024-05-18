# Code Structure
The `multi_node` directory contains the code for running as a separate abstraction layer to SGLang/vLLM in a distributed setting. This code is responsible for coordinating and managing the execution of the distributed system.

## Design Document
For more information about the design and architecture of the `multi_node` code, please refer to the [design document](https://docs.google.com/document/d/1sa0ewITYok7V1kb4tZ32uh6HXitcpcvjybOZpacWKuo/edit).

## Repository Installation
To install the repository, run the following command:

```
pip install -e "python[all]"
pip3 install scipy matplotlib datasets paramiko gurobipy
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.1
```

## Benchmarks

Currently the benchmarking results are all stored in onedrive.
- Add the log file, jupyter notebook for plotting, figure as a seperate marked folder

Run Server
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

Send a request
```
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
```
Learn more about the argument format [here](docs/sampling_params.md).

### OpenAI Compatible API
In addition, the server supports an experimental OpenAI-compatible API.

```python
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# Text completion
response = client.completions.create(
	model="default",
	prompt="The capital of France is",
	temperature=0,
	max_tokens=32,
)
print(response)

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)
```


By default, the server uses the chat template specified in the model tokenizer from Hugging Face. It should just work for most official models such as Llama-2/Llama-3.

If needed, you can also override the chat template when launching the server:

```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --chat-template llama-2
```

If the chat template you are looking for is missing, you are welcome to contribute it.
Meanwhile, you can also temporarily register your chat template as follows:

```json
{
  "name": "my_model",
  "system": "<|im_start|>system",
  "user": "<|im_start|>user",
  "assistant": "<|im_start|>assistant",
  "sep_style": "CHATML",
  "sep": "<|im_end|>",
  "stop_str": ["<|im_end|>", "<|im_start|>"]
}
```

```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --chat-template ./my_model_template.json
```

### Additional Arguments
- Add `--tp 2` to enable tensor parallelism.
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --tp 2
```
- If you see out-of-memory errors during serving, please try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --mem-fraction-static 0.7
```
- You can turn on [flashinfer](docs/flashinfer.md) to accelerate the inference by using highly optimized CUDA kernels.

### Supported Models
- Llama
- Mistral
- Mixtral
- Qwen / Qwen 2
- Gemma
  - Please add a new flag `--attention-reduce-in-fp32` to avoid some precision errors.
  - `python -m sglang.launch_server --model-path google/gemma-7b-it --port 30000 --attention-reduce-in-fp32`
- LLaVA
  - `python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000`
  - `python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-vicuna-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000`
  - `python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-34b --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port 3000`
- Yi-VL
  - see [srt_example_yi_vl.py](examples/quick_start/srt_example_yi_vl.py).
- StableLM
- Command-R
- DBRX
- AWQ/GPTQ/Marlin quantization

Instructions for supporting a new model are [here](https://github.com/sgl-project/sglang/blob/main/docs/model_support.md).

## Benchmark And Performance
- Llama-7B on NVIDIA A10G, FP16, Tensor Parallelism=1
![llama_7b](assets/llama_7b.jpg)

- Mixtral-8x7B on NVIDIA A10G, FP16, Tensor Parallelism=8
![mixtral_8x7b](assets/mixtral_8x7b.jpg)

Learn more [here](docs/benchmark_results.md).

## Roadmap
https://github.com/sgl-project/sglang/issues/157

## Citation And Acknowledgment
```
@misc{zheng2023efficiently,
      title={Efficiently Programming Large Language Models using SGLang},
      author={Lianmin Zheng and Liangsheng Yin and Zhiqiang Xie and Jeff Huang and Chuyue Sun and Cody Hao Yu and Shiyi Cao and Christos Kozyrakis and Ion Stoica and Joseph E. Gonzalez and Clark Barrett and Ying Sheng},
      year={2023},
      eprint={2312.07104},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

We learned from the design and reused some code of the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), [LMQL](https://github.com/eth-sri/lmql).
