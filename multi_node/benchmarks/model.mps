NAME LPTreeTraversal
ROWS
 N  OBJ
 G  R0      
 L  max_per_gpu_cost_constr_0
 L  max_per_gpu_cost_constr_1
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    max_per_gpu_cost  OBJ       1
    max_per_gpu_cost  max_per_gpu_cost_constr_0  -1
    max_per_gpu_cost  max_per_gpu_cost_constr_1  -1
    x_0       OBJ       275.984
    x_0       R0        1
    x_0       max_per_gpu_cost_constr_0  147.4
    x_1       OBJ       275.984
    x_1       R0        1
    x_1       max_per_gpu_cost_constr_1  147.4
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      R0        1
BOUNDS
 LI BND1      max_per_gpu_cost  0
 BV BND1      x_0     
 BV BND1      x_1     
ENDATA
