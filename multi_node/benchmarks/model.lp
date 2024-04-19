\ Model LPTreeTraversal
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  max_per_gpu_cost + 275.984 x_0 + 275.984 x_1
Subject To
 R0: x_0 + x_1 >= 1
 max_per_gpu_cost_constr_0: - max_per_gpu_cost + 147.4 x_0 <= 0
 max_per_gpu_cost_constr_1: - max_per_gpu_cost + 147.4 x_1 <= 0
Bounds
Binaries
 x_0 x_1
Generals
 max_per_gpu_cost
End
