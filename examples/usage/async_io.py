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

    if isinstance(prompt, str):
        prompt = [prompt]
        
    # messages = [[
    #     {"role": "system", "content": "You will be given question answer tasks.",},
    #     {"role": "user", "content": p},
    # ] for p in prompt]

    # prompt = [tokenizer.apply_chat_template(
    #     m, tokenize=False, add_generation_prompt=True
    # ) for m in messages]

    stream = await engine.add_request_await(prompt, sampling_params)
    for response in stream:
        print('-' * 50)
        print(response['text'])
        print('-' * 50)
    # async for output in stream:
    #     print(output, end="", flush=True)
    # print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # runtime = Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
    # runtime = Runtime(model_path="mistralai/Mistral-7B-v0.1", chunk_prefill_budget=512, load_format='dummy')
    runtime = Runtime(model_path="mistralai/Mistral-7B-v0.1", chunk_prefill_budget=512)
    print("--- runtime ready ---\n")
    
    prefix = (
        "You are an expert school principal, skilled in effectively managing "
        "faculty and staff. Draft 10-15 questions for a potential first grade "
        "Head Teacher for my K-12, all-girls', independent school that emphasizes "
        "community, joyful discovery, and life-long learning. The candidate is "
        "coming in for a first-round panel interview for a 8th grade Math "
        "teaching role. They have 5 years of previous teaching experience "
        "as an assistant teacher at a co-ed, public school with experience "
        "in middle school math teaching. Based on these information, fulfill "
        "the following paragraph: ")

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # prompt = ["Who is Alan Turing?", "What is the capital of the United States?", ""]
    # prompt = "What is the capital of the United States?"
    prompts = [prefix + p for p in prompts]
    sampling_params = {"max_new_tokens": 12, 'temperature': 0}
    asyncio.run(generate(runtime, prompts, sampling_params))
    
    runtime.shutdown()
