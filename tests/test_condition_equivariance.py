# Test how the solver equivariance test responds to changes in condition numbers

import numpy as np
from generate_data import get_many_random_pos_poly, transform_many_polys,get_SL2_matrices, induced, get_max_log_det
d = 3
numpolys = 1
nummatrices = 10

ps = get_many_random_pos_poly(numpolys, d) 
ps = np.repeat(ps, nummatrices, axis = 0)
As = get_SL2_matrices(nummatrices, return_tensor=False, mode='gaussian', thresh=10)
Ask = [np.linalg.inv(induced(A, d, scaled = False, return_tensor=False)) for A in As]
ps_transformed = transform_many_polys(ps, As)
M_orig = [get_max_log_det(p, solver = "SCS") for p in ps]
M_transformed = [get_max_log_det(p, solver = "SCS") for p in ps_transformed]
M_retransformed = [Ask[i].T @ np.array(M_transformed[i]) @ Ask[i] for i in range(nummatrices)]
conds = [np.linalg.cond(A) for A in As]
inducedconds = [np.linalg.cond(A) for A in Ask]
errs = np.stack(M_orig) - np.stack(M_retransformed)
errs = np.sum(np.square(errs), axis = (1,2)) / np.linalg.norm(M_orig[0])

# For plotting
import pandas as pd
df = pd.DataFrame({'conds':conds, 'inducedconds':inducedconds,'errs':errs})