name_sys = 'Name'
name_new_sys = 'OurPolicy'
name_new_sys_unfair = 'PolicyWithoutFairness'
name_sglang = 'SGLang'
name_vllm = 'vLLM'
name_oracle = 'Consistent Hashing'

MARKER_SIZE = 3

line_sys = {'color': '#2ca02c', 'label': name_sys, 'marker': 'v', 'markersize': MARKER_SIZE}
line_sys_without_fairness = {'color': '#2ca02d', 'label': name_new_sys_unfair, 'marker': 'v', 'markersize': MARKER_SIZE}
line_new_sys = {'color': '#1f77b4', 'label': name_new_sys, 'marker': 's', 'markersize': MARKER_SIZE}
line_sglang = {'color': '#ff7f0e', 'label': name_sglang, 'marker': 'o', 'markersize': MARKER_SIZE}
line_vllm = {'color': '#EA4336', 'label': name_vllm, 'marker': 'x', 'markersize': MARKER_SIZE}
line_oracle = {'color': '#8c564b', 'label': name_oracle, 'marker': '^', 'markersize': MARKER_SIZE}

policy_mapping = {
    'ROUND_ROBIN:': line_sglang,
    'CUSTOM:GlobalScheduler': line_sys,

    'CUSTOM:GlobalSchedulerTime': line_sys,
    'CUSTOM:GlobalSchedulerTimeWithEvictionWithLPM': line_sys_without_fairness,

    'CUSTOM:GlobalSchedulerWithoutMissRate': line_sys,
    'CUSTOM:GlobalSchedulerTimeWithEviction': line_new_sys,
    'CUSTOM:ORACLE': line_oracle,
}

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 11})

from typing import Dict, List, Optional, Iterator
import numpy as np
import pandas as pd
import re

def read_e2e_csv_metrics(fpaths: List[str]):
    if isinstance(fpaths, str):
        fpaths = [fpaths]
    dataframes = [pd.read_csv(file) for file in fpaths]
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Function to extract rps from the experiment_id column
    def extract_rps(experiment_id):
        match = re.search(r'rps=(.+?),', experiment_id)
        return float(match.group(1)) if match else None
    
    # Apply the function to the experiment_id column and create a new column for rps
    if 'rps' not in combined_df.columns:
        combined_df['rps'] = combined_df['experiment_id'].apply(extract_rps)

    combined_df.fillna({'custom_policy': ''}, inplace=True)
    combined_df.fillna({'custom_policy_msg': ''}, inplace=True)

    # print(combined_df[['experiment_id', 'rps']])
    combined_df = combined_df.sort_values(by='rps')
    grouped = combined_df.groupby(['policy', 'custom_policy', 'custom_policy_msg'])
    return grouped