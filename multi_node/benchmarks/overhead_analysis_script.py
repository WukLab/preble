#%%
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load JSON data
file_name = 'metrics_lp_scheduler.json'
with open(file_name, 'r') as file:
    data = json.load(file)

# Extract overhead values
overhead_values = [entry['overhead'] for entry in data]
max_index = np.argmax(overhead_values)
max_value = overhead_values[max_index]
print("Index with max overhead:", max_index, data[max_index]['text'])
print("Value with max overhead:", max_value)

std_dev = np.std(overhead_values) / 4  # Reduced for visualization purposes
error = [std_dev] * len(overhead_values)
# Initial Plot
plt.figure(figsize=(10, 6))
plt.plot(overhead_values, marker='o', linestyle='-', color='b')
plt.title('Initial Overhead over Time')
plt.xlabel('Time (sequential order of metrics)')
plt.ylabel('Overhead (seconds)')
plt.grid(True)
# plt.savefig('initial_overhead_plot.png')
# plt.close()
plt.show()
# More Smoothed Plot
more_smoothed_overhead = savgol_filter(overhead_values, 21, 3)
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(overhead_values)), more_smoothed_overhead, yerr=error, marker='o', linestyle='-', color='g', ecolor='lightgray', elinewidth=3, capsize=0)
plt.title('More Smoothed Overhead over Time with Error Bars')
plt.xlabel('Time (sequential order of metrics)')
plt.ylabel('Overhead (seconds)')
plt.grid(True)
# plt.savefig('more_smoothed_overhead_with_error_bars.png')
# plt.close()
plt.show()
# %%
waiting_queue_len_values = [entry['metrics'][0]['waiting_queue_len'] for entry in data]  # Assuming first metric is representative
def running_average(data, window_size):
    """Compute the running average with a specified window size."""
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

waiting_queue_len_values_smoothed = running_average(waiting_queue_len_values, 5)
# For demonstration, let's overlay this with the more smoothed overhead plot,
# reusing the 'more_smoothed_overhead' and 'error' variables from earlier.

# Re-plotting the more smoothed overhead with error bars
fig, ax1 = plt.subplots(figsize=(10, 6))
import numpy as np

# Overhead plot on ax1
color = 'tab:green'
ax1.set_xlabel('Time (sequential order of metrics)')
ax1.set_ylabel('Overhead (seconds)', color=color)
ax1.errorbar(range(len(overhead_values)), more_smoothed_overhead, yerr=error, marker='o', linestyle='-', color=color, ecolor='lightgray', elinewidth=3, capsize=0, label='Overhead (smoothed)')
ax1.tick_params(axis='y', labelcolor=color)

# Create a twin axis for the waiting queue length
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Waiting Queue Length', color=color)
ax2.plot(waiting_queue_len_values_smoothed, marker='x', linestyle='--', color=color, label='Waiting Queue Length')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Overhead and Waiting Queue Length over Time')
plt.savefig('overhead_and_waiting_queue_length_correlation.png')
plt.show()

# %%
# Initialize a dictionary to map the first 10 characters to the selected runtime
runtime_selection = {}
correct_predictions = 0
ignore_items = 0
import re

for entry in data:
    text = entry['text']
    match = re.search(r"You have access of the following tools:\n1.(.+?): ", text)
    if match:
        tool = match.group(1)
    first_10_chars = tool
    selected_runtime = entry['selected_runtime']
    if "Workload" not in first_10_chars:
        ignore_items += 1
        continue
    # If the first 10 characters have been seen before
    if first_10_chars in runtime_selection:
        # Check if the selected runtime matches the previously recorded runtime
        if runtime_selection[first_10_chars] == selected_runtime:
            correct_predictions += 1
    else:
        # Record the runtime selection for the first 10 characters
        runtime_selection[first_10_chars] = selected_runtime
        # Assuming the first selection is always correct as there's no precedent
        correct_predictions += 1

# Calculate the accuracy
accuracy_percentage = (correct_predictions / (len(data) - ignore_items)) * 100
accuracy_percentage

# %%
# Initialize lists to store data for plotting
total_radix_cache_times = []
tokenization_times = []
queue_processing_times = []

# Extract relevant metrics from the JSON data and aggregate
for entry in data:
    for metric in entry['metrics']:
        total_radix_cache_times.append(metric['total_radix_cache_processing_time'])
        tokenization_times.append(metric['tokenization_time'])
        queue_processing_times.append(metric['queue_processing_time'])

# Calculate mean and standard deviation for each metric
total_radix_cache_mean = np.mean(total_radix_cache_times)
tokenization_mean = np.mean(tokenization_times)
queue_processing_mean = np.mean(queue_processing_times)

total_radix_cache_std = np.std(total_radix_cache_times)
tokenization_std = np.std(tokenization_times)
queue_processing_std = np.std(queue_processing_times)

# Plotting
metrics = ['Total Radix Cache Processing Time', 'Tokenization Time', 'Queue Processing Time']
means = [total_radix_cache_mean, tokenization_mean, queue_processing_mean]
stds = [total_radix_cache_std, tokenization_std, queue_processing_std]

plt.figure(figsize=(10, 6))

# Plot means with error bars representing standard deviation
plt.bar(metrics, means, yerr=stds, capsize=10)
plt.title('Aggregated Metrics Across Workloads')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# %%
