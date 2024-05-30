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
            # if "ORACLE" in name[1]:
            #     name = ('CUSTOM', 'ORACLE', '')
            # if "ROUND_ROBIN" in name[0]:
            #     name = ('ROUND_ROBIN', '', '')
            # ax[i].plot(group['rps'], group[y_column], **plot_utils.policy_mapping[':'.join(name)])
            ax[i].plot(group['rps'], group[y_column], label = [':'.join(name)], marker = 'o')
            ax[i].set_xlabel('RPS')
        ax[i].legend()
        ax[i].set_title(y_column)
        if x_lim:
            ax[i].set_xlim(x_lim)
                
fig, ax = plt.subplots(1, 2, figsize=(16.5, 9.5))
y_columns = ['average_request_latency', 'p99_latency']
# plot_all_metrics([
#     '/mnt/ssd1/alm-os/sglang_multi_model/ckpt_all_in_one/2r_loogle_k=10/exp.csv', 
#     '/mnt/ssd1/alm-os/sglang_multi_model/ckpt_all_in_one/2r_loogle_k=10_rps>0.7/exp.csv', 
#     '/mnt/ssd1/alm-os/sglang_multi_model/ckpt_all_in_one/2r_loogle_lpm_ours/exp.csv'
#                   ], ax, y_columns)
# plot_all_metrics('/mnt/ssd1/alm-os/sglang_multi_model/ckpt_all_in_one/2r_loogle_lpm_ours/exp.csv', ax, y_columns)
# plot_all_metrics(['/mnt/ssd1/alm-os/sglang_multi_model/fix_wq_real/2r_loogle_new_data/exp.csv'], ax, y_columns)
# plot_all_metrics(['/mnt/ssd1/alm-os/sglang_multi_model/fix_wq/2r_loogle_new_data/exp.csv'], ax, y_columns)
plot_all_metrics(['/mnt/ssd1/alm-os/sglang_multi_model/final_fix_wq_sim/2r_loogle/exp.csv'], ax, y_columns)
plt.tight_layout()
plt.savefig('easy_plot.png')