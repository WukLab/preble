# Preble

Preble is a load balancer for effecient prefix caching systems
## Installation

You can install the package using pip:

# Code Structure
The `multi_node` directory contains the code for running as a separate abstraction layer to SGLang/vLLM in a distributed setting. This code is responsible for coordinating and managing the execution of the distributed system.

Editable Installation
```
pip3 install -e .
pip install -e "python[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```

Regular Pip Installation:
```
pip3 install preble
pip install git+https://github.com/wuklab/preble.git#egg=preble[all]
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```


We release a custom version of sglang that supports chunked prefill

## Programatically starting the server
We can support providing a list of runtime urls
```
from preble.main import start_server

start_server(
    runtime_selection_policy="custom",
    runtime_urls="http://127.0.0.1:30000/generate,http://127.0.0.1:30001/generate",
    host='127.0.0.1',
    port=8000,
    model="mistralai/Mistral-7B-v0.1"
)
```

We can also support dynamically loading the models to seperate cuda devices
```
from preble.main import start_server_and_load_models

start_server_and_load_models(
    model_name="mistralai/Mistral-7B-v0.1",
    devices=[0, 1],
    host="127.0.0.1",
    port=8000
)
```

The server can be run via:
```
python3 multi_node/server/server.py <server/deploy_and_run>
```
- server runs the server given a list of urls
- deploy_and_run generates two endpoints

CLI Configuration
```
    runtime_selection_policy: The policy to select the runtime (e.g., custom, round_robin).
    runtime_urls: Comma-separated list of runtime URLs.
    host: The host address for the server.
    port: The port number for the server.
    model: The model to be used (e.g., mistralai/Mistral-7B-v0.1).
```

## Citation And Acknowledgment
The code is forked of sglang

License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.