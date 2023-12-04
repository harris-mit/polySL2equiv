import numpy as np
from sympy.functions.special.polynomials import gegenbauer
import cvxpy 
import argparse

import pickle
import sympy
import time

from generate_sphere_packing_data import solve_spherical_code
from generate_data import get_many_random_pos_poly, get_max_log_det

def main(args: argparse) -> None:
    k = int(args.degree / 2)
    all_d = []
    all_alpha = []
    all_polys = []
    all_certs = []
    numtrials = args.num_trials
    start = time.time()
    for d in range(3, k+1):
        print(f'ELAPSED TIME at start of d={d}', (time.time() - start)/60, 'minutes')
        for i in range(numtrials):
            alpha = (np.random.rand()*2) - 1
            # k must be >= d
            # the answer is degree 2*k (i.e. length 2*k + 1) 
            try:
                out = solve_spherical_code(k, alpha, d, solver = "SCS")
                
                out = -1 * out
                out = out + 1e-4
                
                try:
                    mat = get_max_log_det(out, solver="SCS").float()
                    print(f'solve succeeded: k {k} d {d} alpha {alpha:.2f}')
                    all_polys.append(out)
                    all_certs.append(mat)
                    all_d.append(d)
                    all_alpha.append(alpha)
                except:
                    print(f'SOLVE FAILED: k {k} d {d} alpha {alpha:.2f}')
                
            except:
                print(f'POLY GENERATION FAILED: k {k} d {d} alpha {alpha:.2f}')
    outdict = {'k': k, 'all_d': all_d, 'all_alpha': all_alpha, 'all_polys': all_polys, 'all_certs': all_certs}

    with open(f'sphere_deg_{args.degree}_trials_{args.num_trials}.pkl', 'wb') as f:
        pickle.dump(outdict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate sphere polynomial dataset')

    # General training arguments
    parser.add_argument('--degree', type=int, default=10,
                        help='Degree (i.e. k/2)')
    
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Degree (i.e. k/2)')
    
    args = parser.parse_args()
    main(args)
    

