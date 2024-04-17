"""
Usage:
python3 async_io.py
"""
import asyncio
from sglang import Runtime
import logging


async def generate(
    engine,
    prompt,
    sampling_params,
):
    tokenizer = engine.get_tokenizer()

    messages = [
        {"role": "system", "content": "You will be given question answer tasks.",},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    stream = engine.add_request(prompt, sampling_params)

    async for output in stream:
        print(output, end="", flush=True)
    print()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    # runtime = Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
    runtime = Runtime(model_path="mistralai/Mistral-7B-v0.1", chunk_prefill_budget=0)
    print("--- runtime ready ---\n")

    prompt = "Who is Alan Turing?"
    sampling_params = {"max_new_tokens": 128, 'temperature': 1.0}
    asyncio.run(generate(runtime, prompt, sampling_params))
    
    runtime.shutdown()
