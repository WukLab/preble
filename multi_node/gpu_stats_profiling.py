from threading import Thread
from pynvml.smi import nvidia_smi
import time
import pandas as pd
import psutil
import logging

log = logging.getLogger(__name__)


def get_gpu_profile():
    smi_instance = nvidia_smi.getInstance()
    gpu_stats = smi_instance.DeviceQuery("memory.free, memory.total, utilization.gpu")[
        "gpu"
    ]
    gpu_stats_dic = {}
    for index, gpu_stat in enumerate(gpu_stats):
        memory_usage = (
            1
            - gpu_stat["fb_memory_usage"]["free"] / gpu_stat["fb_memory_usage"]["total"]
        )
        gpu_usage = gpu_stat["utilization"]["gpu_util"]
        gpu_stats_dic[index] = {"memory_usage": memory_usage, "gpu_usage": gpu_usage}
    return gpu_stats_dic


class Monitor(Thread):
    def __init__(self, delay=0.1, save_path=None):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.stats = []
        self.start_time = time.time()
        self.smi_instance = nvidia_smi.getInstance()
        self.save_path = save_path
        self.start()

    def run(self):
        while not self.stopped:
            # GPUtil.showUtilization()
            curr_time = time.time() - self.start_time
            gpu_stats = self.smi_instance.DeviceQuery(
                "memory.free, memory.total, utilization.gpu"
            )["gpu"]
            for index, gpu_stat in enumerate(gpu_stats):
                memory_usage = (
                    1
                    - gpu_stat["fb_memory_usage"]["free"]
                    / gpu_stat["fb_memory_usage"]["total"]
                )
                gpu_usage = gpu_stat["utilization"]["gpu_util"]

                self.stats.append(
                    {
                        "gpu_id": index,
                        "gpu_memory": memory_usage,
                        "gpu_mem_total": gpu_stat["fb_memory_usage"]["total"],
                        "gpu_usage": gpu_usage,
                        "elapsed_time": curr_time,
                        "cpu": psutil.cpu_percent(),
                        "cpu_mem": psutil.virtual_memory().percent,
                    }
                )
            time.sleep(self.delay)

    def log_aggregate_stats(self) -> str:
        if len(self.stats) == 0:
            return "No Stats"
        grouped_stats = pd.DataFrame(self.stats).groupby("gpu_id")

        # Iterate over each GPU group
        for gpu_id, gpu_group in grouped_stats:
            # Calculate mean and standard deviation for load and memory
            mean_load = gpu_group["gpu_usage"].mean()
            std_load = gpu_group["gpu_usage"].std()
            mean_memory = gpu_group["gpu_memory"].mean()
            std_memory = gpu_group["gpu_memory"].std()

            output_stats = (
                f"GPU {gpu_id}: "
                f"GPU Usage: {mean_load:.2f}±{std_load:.2f}, "
                f"Memory: {mean_memory:.2f}±{std_memory:.2f}, "
            )
            log.debug(output_stats)

    def stop(self):
        self.stopped = True
        if self.save_path is not None:
            pd.DataFrame(self.stats).to_csv(self.save_path)


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        monitor = Monitor(0.1)
        result = func(*args, **kwargs)
        end_time = time.time()
        monitor.stop()
        print(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute."
        )
        monitor.log_aggregate_stats()
        return result

    return wrapper


def async_timeit(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        monitor = Monitor(0.1)
        result = await func(*args, **kwargs)
        end_time = time.time()
        monitor.stop()
        print(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute."
        )
        monitor.log_aggregate_stats()
        return result

    return wrapper


if __name__ == "__main__":
    import cupy as cp

    # show logging debug
    logging.basicConfig(level=logging.DEBUG)

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

    test_gpu_load(1000)


# Test eviction rate/hit rate for radix cache
#     larger weight usage 80% for cache. limit the kv cache further
#         normal decoding vs decoding (prefix vs others)
#         full throughput vs without cache
#         roofline plot
#         co-location of two machines
#             each service has multiple models(Better binpacking)
#             Communication cost is lower
