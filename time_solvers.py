from generate_data import get_many_random_pos_poly, get_max_log_det
import numpy as np
import time
import torch

# only run on a few examples, and just generate them here?

degree_range = [6, 8, 10, 12, 14] #range(4,32,2) #[4, 8, 12, 16, 20, 24, 28, 32]
data = {}
times_scs = {}
times_mosek = {}
errors = {}
numsamples = 100
for deg in degree_range:
    #try:
    data[deg] = get_many_random_pos_poly(numsamples, int(deg / 2))
    times_scs[deg] = []
    times_mosek[deg] = []
    errors[deg] = []
    start = time.time()
    for i in range(numsamples):
        x = data[deg][i]
        start = time.time()
        M_mosek = get_max_log_det(x, solver='MOSEK')
        times_mosek[deg] += [time.time() - start]
        start = time.time()
        M_scs = get_max_log_det(x, solver='SCS')
        times_scs[deg] += [time.time() - start]
        errors[deg] += [torch.norm(M_scs - M_mosek).item() / (torch.norm(M_mosek) + 1).item()]
    
    print('Degree ', deg, ' time (m) for 5k mosek ', sum(times_mosek[deg])/numsamples*5000/60,
          ' time (m) for 5k scs ', sum(times_scs[deg])/numsamples*5000/60, ' avg error ', sum(errors[deg])/numsamples)
