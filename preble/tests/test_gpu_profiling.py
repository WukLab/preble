import sys
import os

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpu_stats_profiling import Monitor

import unittest
import time
import cupy as cp


def test_gpu_load(matrix_size, iterations=500):
    # Generate random matrices on GPU
    monitor = Monitor(0.1)
    for _ in range(iterations):
        matrix_a = cp.random.rand(matrix_size, matrix_size)
        matrix_b = cp.random.rand(matrix_size, matrix_size)

        # Measure the time taken for matrix multiplication
        result = cp.dot(matrix_a, matrix_b)
    monitor.stop()
    monitor.log_aggregate_stats()
    return result


class TestMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = Monitor(delay=0.1)

    def tearDown(self):
        self.monitor.stop()

    def test_monitor_run(self):
        time.sleep(1)  # Let the monitor run for 1 second
        self.assertGreater(len(self.monitor.stats), 10)

    def test_monitor_log_aggregate_stats(self):
        # Ensure that the log_aggregate_stats function runs without errors
        self.monitor.log_aggregate_stats()

    def test_monitor_with_load_test(self):
        matrix_size = 1000
        result = test_gpu_load(matrix_size)
        self.assertIsNotNone(result)


# TODO test timeit decorator
if __name__ == "__main__":
    unittest.main()
