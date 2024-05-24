import requests
import json
payload = {
    "stream": True,
    "text": "What is the capital of san jose? Generate me a 500 word essay on it",
    "input_ids": [1, 2, 3],
    "sampling_params": {
        "max_new_tokens": 200
    }
}
import requests
from typing import Iterable, List

url = "http://127.0.0.1:8000/process"  # Replace with your URL

headers = {
    'Accept': 'text/event-stream',
    'Content-Type': 'application/json',  # Adjust if needed
}

def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)

def process_event(event):
    print(f"Received event: {event}")
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=24,
                                     decode_unicode=False
                                     ):
        if chunk:
            chunk = remove_prefix(chunk.decode("utf-8"), "data:").strip()
            if chunk == "[DONE]":
                break
            data = json.loads(chunk)
            output = data["text"]
            yield output

response = requests.post(url, json=payload, headers=headers, stream=True)
num_printed_lines = 0
for h in get_streaming_response(response):
    # clear_line(num_printed_lines)
    num_printed_lines = 0
    for i, line in enumerate(h):
        num_printed_lines += 1
        line = line.strip("\n")
        if line:
            print(line, end='')
    