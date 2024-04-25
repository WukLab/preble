from histogram_based_scheduling_v2 import SlidingWindowHistogram, LPTreeNode, HistogramBasedRecompV2, TTFTWindowedOverloadedDetector
import unittest
from datetime import datetime, timedelta

class TestGPULoadBalancing(unittest.TestCase):
    def _create_node(self, load, num_tokens, gpus, histogram, per_gpu_load: dict, gpu_allocations: dict):
        node = LPTreeNode()
        node.load = load
        node.ref_counter = load
        node.value = [0 for _ in range(num_tokens)]
        node.context_length = len(node.value)
        node.cached_gpus = gpus
        gpu_allocations[node] = gpus
        for i in range(load):
            histogram.update(datetime.now(), node, node)
            for gpu in gpus:
                per_gpu_load[gpu] += 1
        return node

    def test_basic_balance_transfer(self):
        # 2 gpus. One has higher load than the other
        num_gpus = 2
        per_gpu_load = {i: 0 for i in range(num_gpus)}
        gpu_allocations = {}
        histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), gpu_allocations=gpu_allocations, num_gpus=num_gpus)

        scheduler = HistogramBasedRecompV2(num_nodes=2, enable_eviction=False)
        node0_gpu0 = self._create_node(load=50, num_tokens=50, gpus={0}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)
        node1_gpu0 = self._create_node(load=5,  num_tokens=10, gpus={0}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)
        node2_gpu1 = self._create_node(load=20,  num_tokens=10, gpus={1},histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)
        
        scheduler.gpu_allocations = gpu_allocations
        scheduler.per_gpu_load = per_gpu_load
        scheduler.histogram = histogram
        # Call your method to test
        scheduler.handle_important_node_stealing(0)
        self.assertEquals({
            node0_gpu0: {0},
            node1_gpu0: {1},
            node2_gpu1: {1}
        }, scheduler.gpu_allocations)

    def test_no_transfer_when_loads_are_close(self):
        num_gpus = 2
        per_gpu_load = {i: 0 for i in range(num_gpus)}
        gpu_allocations = {}
        histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), gpu_allocations=gpu_allocations, num_gpus=num_gpus)

        scheduler = HistogramBasedRecompV2(num_nodes=2, enable_eviction=False)
        node0_gpu0 = self._create_node(load=25, num_tokens=50, gpus={0}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)
        node1_gpu1 = self._create_node(load=20, num_tokens=10, gpus={1}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)

        scheduler.gpu_allocations = gpu_allocations
        scheduler.per_gpu_load = per_gpu_load
        scheduler.histogram = histogram
        scheduler.handle_important_node_stealing(0)
        self.assertEquals({
            node0_gpu0: {0},
            node1_gpu1: {1}
        }, scheduler.gpu_allocations)

    def test_splitting_a_very_large_prefix_on_one_gpu(self):
        num_gpus = 2
        per_gpu_load = {i: 0 for i in range(num_gpus)}
        gpu_allocations = {}
        histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), gpu_allocations=gpu_allocations, num_gpus=num_gpus)

        scheduler = HistogramBasedRecompV2(num_nodes=2, enable_eviction=False)
        node0_gpu0 = self._create_node(load=62, num_tokens=50, gpus={0}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)
        node1_gpu1 = self._create_node(load=10, num_tokens=10, gpus={1}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)
        
        scheduler.gpu_allocations = gpu_allocations
        scheduler.per_gpu_load = per_gpu_load
        scheduler.histogram = histogram
        scheduler.handle_important_node_stealing(0)
        # Assuming the load was meant to be balanced by the scheduler
        # No stealing bc not overloaded

        self.assertEquals(scheduler.gpu_allocations, {
            node0_gpu0: {0},
            node1_gpu1: {1}
        })
        overloaded_detector = TTFTWindowedOverloadedDetector(timedelta(minutes=3))
        # Shows the TTFT blowing up/doubling
        overloaded_detector.add_data_point(datetime.now() - timedelta(minutes=2), node=node0_gpu0, gpu=0, value=0.04)
        overloaded_detector.add_data_point(datetime.now() - timedelta(minutes=2), node=node0_gpu0, gpu=0, value=0.04)
        overloaded_detector.add_data_point(datetime.now() - timedelta(minutes=2), node=node0_gpu0,  gpu=0, value=0.04)
        overloaded_detector.add_data_point(datetime.now() - timedelta(minutes=2), node=node0_gpu0,  gpu=0, value=0.04)

        overloaded_detector.add_data_point(datetime.now(), node=node0_gpu0, gpu=0, value=0.085)
        overloaded_detector.add_data_point(datetime.now(), node=node0_gpu0, gpu=0, value=0.085)
        overloaded_detector.add_data_point(datetime.now(), node=node0_gpu0, gpu=0, value=0.085)
        overloaded_detector.add_data_point(datetime.now(), node=node0_gpu0, gpu=0, value=0.085)
        scheduler.overload_detector = overloaded_detector

        scheduler.handle_important_node_stealing(0)
        # Monkey patch scheduler.overload_detector.
        self.assertEquals(scheduler.gpu_allocations, {
            node0_gpu0: {0, 1},
            node1_gpu1: {1}
        })

    def test_multiple_nodes_with_cascading_transfers(self):
        num_gpus = 3
        per_gpu_load = {i: 0 for i in range(num_gpus)}
        gpu_allocations = {}
        histogram = SlidingWindowHistogram(window_duration=timedelta(minutes=3), gpu_allocations=gpu_allocations, num_gpus=num_gpus)

        scheduler = HistogramBasedRecompV2(num_nodes=3, enable_eviction=False)

        node0_gpu0 = self._create_node(load=50, num_tokens=50, gpus={0}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)
        node1_gpu0 = self._create_node(load=10, num_tokens=20, gpus={0}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)

        node2_gpu1 = self._create_node(load=50, num_tokens=50, gpus={1}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)
        node3_gpu1 = self._create_node(load=10, num_tokens=20, gpus={1}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)

        node4_gpu2 = self._create_node(load=5, num_tokens=10, gpus={2}, histogram=histogram, per_gpu_load=per_gpu_load, gpu_allocations=gpu_allocations)

        scheduler.gpu_allocations = gpu_allocations
        scheduler.per_gpu_load = per_gpu_load
        scheduler.histogram = histogram
        scheduler.handle_important_node_stealing(0)
        # Ideally node 1, node3 should be moved to node 4
        self.assertEquals({
            node0_gpu0: {0},
            node1_gpu0: {2},
            node2_gpu1: {1},
            node3_gpu1: {2},
            node4_gpu2: {2}
        }, scheduler.gpu_allocations)

        scheduler.handle_important_node_stealing(0)

        self.assertEquals({
            node0_gpu0: {0},
            node1_gpu0: {2},
            node2_gpu1: {1},
            node3_gpu1: {2},
            node4_gpu2: {2}
        }, scheduler.gpu_allocations)

        # After another iteration, the allocation remains stable/no splitting

    

if __name__ == '__main__':
    unittest.main()