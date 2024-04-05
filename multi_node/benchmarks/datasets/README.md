# Getting relevant datasets

Share GPT Dataset: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json

Toolbench: dataset based on https://github.com/OpenBMB/ToolBench G1_workload.json. Then we pre-process it to remove any outliers. This dataset also accidently mixes two versions of the sample input prompt.

Currently, the easiest way to obtain it is to copy it over from the server at 
`/mnt/data/ssd/sglang_multi_model/multi_node/benchmarks/G1_workload_updated_with_input_output.json`
