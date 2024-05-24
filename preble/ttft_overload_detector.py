from collections import deque
from datetime import datetime, timedelta

class TTFTWindowedOverloadedDetector:
    # TTFT is a good indicator of overloaded

    def __init__(self, window_duration=timedelta(minutes=3)):
        self.data = {}
        self.window_duration = window_duration
        self.half_window_duration = window_duration / 2

    def add_data_point(self, timestamp, node, gpu, value):
        """ Add a new data point and remove outdated entries. """
        key = (node, gpu)
        if key not in self.data:
            self.data[key] = deque()
        self.data[key].append((timestamp, value))
        self.purge_old_data(key, timestamp)

    def purge_old_data(self, key, current_time):
        """ Remove data points that are older than the time window. """
        while self.data[key] and self.data[key][0][0] < current_time - self.window_duration:
            self.data[key].popleft()

    def rename_node(self, old_node, new_node, runtime_idx):
        old_key = (old_node, runtime_idx)
        new_key = (new_node, runtime_idx)
        if old_key in self.data:
            self.data[new_key] = self.data.pop(old_key)

    def calculate_half_window_averages(self, key):
        """ Calculate averages for the first and second halves of the window. """
        first_half = []
        second_half = []
        half_window_cutoff = datetime.now() - self.half_window_duration
        if key not in self.data:
            return None, None
        for timestamp, value in self.data[key]:
            if timestamp < half_window_cutoff:
                first_half.append(value)
            else:
                second_half.append(value)
        if len(first_half) == 0:
            return None, None
        if len(second_half) == 0:
            return  None, None
        avg_first_half = sum(first_half) / len(first_half) if first_half else 0
        avg_second_half = sum(second_half) / len(second_half) if second_half else 0

        return avg_first_half, avg_second_half
    
    def delete_after_allocation(self, node, gpu):
        key = (node, gpu)
        if key in self.data:
            del self.data[key]

    def is_node_overloaded(self, node, gpu):
        """ Check if node is overloaded """
        key = (node, gpu)
        avg_first_half, avg_second_half = self.calculate_half_window_averages(key)
        if avg_first_half is None and avg_second_half is None:
            return False
        return avg_second_half >= 2 * avg_first_half
