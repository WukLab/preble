import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plot_utils
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator


def plot_all_metrics(fpath, ax, y_columns, x_lim=None):
    df = plot_utils.read_e2e_csv_metrics(fpath)
    for i, y_column in enumerate(y_columns):
        for name, group in df:
            if "ORACLE" in name[1]:
                name = ('CUSTOM', 'ORACLE')
            ax[i].plot(group['rps'], group[y_column], **plot_utils.policy_mapping[':'.join(name)])
            ax[i].set_xlabel('RPS')
            ax[i].legend()
            ax[i].set_title(y_column)
            if x_lim:
                ax[i].set_xlim(x_lim)
                
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
y_columns = ['average_request_latency', 'p99_latency']
plot_all_metrics('/home/exx/nsdi_zijian/stateful_llm_serving/real_ckpt_all_in_one/2r_loogle_H100_final_ours/exp.csv', ax, y_columns)
# plot_all_metrics('/mnt/ssd1/alm-os/sglang_multi_model/real_ckpt_all_in_one/3r_toolbench/exp.csv', ax, y_columns2)
fig.suptitle('ToolBench 70B 2x4 GPUs')
plt.tight_layout()
plt.savefig('tb-70B.png')