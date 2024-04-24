def get_recomp_cost(self, node: LPTreeNode, gpu_id, histogram: SlidingWindowHistogram):
    if not node or gpu_id in node.gpu_selections:
        return 0
    else:
        return node.num_tokens * histogram.histogram.get(node, 0) + self.get_recomp_cost(node.parent, gpu_id, histogram)

def runtime_selector(
    input_ids
):
    split_nodes = {}
    leaf_node = self.cache.insert(tuple(input_ids))
    heavy_node = self.get_heavy_node(leaf_node)
    if leaf_node.num_tokens < leaf_node.context_length - leaf_node.num_tokens:
        gpu_selected = self.get_parent_gpu_selections(leaf_node)
    else:
        recom_costs = []
        for gpu_id in range(self.num_gpus):
            recomputation_cost = self.get_recomp_cost(leaf_node.parent, gpu_id, self.hisotgram)
            recom_costs.append(recomputation_cost)
        histogram_mem_cost = self.hisotgram.current_allocation_per_gpu()
        gpu_selected = int(np.argmin([recom_costs[gpu_id] + histogram_mem_cost[gpu_id] for gpu_id in range(self.num_gpus)]))
        self.mem_cost[gpu_selected] += leaf_node.context_length
        gpu_selected = set([gpu_selected])
    self.update_gpu_selections_of_parent(leaf_node, gpu_selected)
    self.hisotgram.update(datetime.now(), heavy_node, leaf_node)
    if len(gpu_selected) > 1:
        random_gpu = np.random.choice(list(gpu_selected))
        return int(random_gpu)
    runtime_idx = list(gpu_selected)[0]
    return int(runtime_idx)
