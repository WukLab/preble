import requests
url = "http://127.0.0.1:30000"
headers = {}
res = requests.post(
    url + "/generate",
    json={
        "text": "Say this is a warmup request.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16,
        },
    },
    headers=headers,
    timeout=60,
)

print(res)