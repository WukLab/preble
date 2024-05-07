sglang_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'lpm',
    # "chunk_prefill_budget": 512,
}

ours_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.75,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'fcfs-mpq',
    "chunk_prefill_budget": 1024,
    'report_hit_ratio': True ,
    'enable_iterative_eviction': False,
    'enable_partial_eviction': True,
}

def kwargs_to_cli_args(**kwargs):
    args = []
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key.replace('_', '-')}")
        else:
            args.append(f"--{key.replace('_', '-')} {value}")
    return ' '.join(args)

print(kwargs_to_cli_args(**ours_server_args))

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-70B --load-format dummy --host 0.0.0.0 --log-prefix-hit --mem-fraction-static 0.75 --context-length 32768 --enable-flashinfer --schedule-heuristic lpm --tp-size 4  --port 2334

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-70B --load-format dummy --host 0.0.0.0 --log-prefix-hit --mem-fraction-static 0.75 --context-length 32768 --enable-flashinfer --schedule-heuristic fcfs-mpq --chunk-prefill-budget 1024 --report-hit-ratio --enable-partial-eviction --tp-size 4 --port 2333




# point switch to swap
# expand the x and show AIFM
