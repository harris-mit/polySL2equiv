import numpy as np
import math
import torch
import sys
sys.path.append('tests')
from generate_data import get_many_random_pos_poly, get_SL2_matrices, induced, transform_poly_dict, get_rotations
import argparse
import os
import pickle
from tqdm import tqdm
import time

def main(args: argparse):
    start = time.time()
    os.makedirs(args.save_dir, exist_ok=True)
    print('Computing 2x2 matrices...')
    if args.rotations:
        As = get_rotations(args.num, return_tensor=True)
    else:
        if args.cond_range is None:
            As = get_SL2_matrices(args.num, return_tensor=True, mode=args.transform_mode, thresh=args.thresh, cond_range=None)
        else:
            As, cond_nums, cond_ids = get_SL2_matrices(args.num, return_tensor=True, mode=args.transform_mode, thresh=args.thresh, cond_range=args.cond_range)
    lst = []
    print('Done computing 2x2 matrices')
    for i in tqdm(range(As.shape[0])): 
        A = As[i]
        induced_mats = []
        for k in range(args.max_degree + 1): # max degree or something else? no I think degree is right 
            induced_deg_k = induced(A, k=k, scaled=False, return_tensor=True)
            induced_mats.append(induced_deg_k)
        lst.append(induced_mats)
    end = time.time()
    print(f'Elapsed time: {(end-start)/60} minutes')
    if args.cond_range is None:
        fname = os.path.join(args.save_dir, args.save_name + '.pkl')
        print(f'Writing to {fname}')
        with open(fname, 'wb') as f:
            pickle.dump({'As': As, 'induced': lst}, f)
    else:
        for i in range(len(args.cond_range) - 1):
            fname = os.path.join(args.save_dir, f'{args.save_name}_lower_{args.cond_range[i]}_upper_{args.cond_range[i+1]}.pkl')
            print(f'Writing to {fname}')
            relevant_inds = torch.where(cond_ids == i)[0]
            with open(fname, 'wb') as f:
                pickle.dump({'As': As[relevant_inds], 'induced': [lst[i] for i in relevant_inds]}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute induced matrices of SL2')

    parser.add_argument('--save_dir', type=str, default='/home/user/TransvectantNets_shared/precomputations/induced',
                        help='--Directory in which to save precomputed induced matrices.')
    parser.add_argument('--save_name', type=str, default='most_recent',
                        help='--File name for saving precomputed induced matrices.')
    parser.add_argument('--rotations', default=False, action='store_true', help='--If included, only compute rotation matrices, not general elements of SL2.')
    parser.add_argument('--num', type=int, default=10000, help='--Number to save')
    parser.add_argument('--max_degree', type=int, default=1000, help='--Max degree of induced matrix to save')
    parser.add_argument('--transform_mode', type=str, default='gaussian',
                        help='gaussian or iwasawa as distribution over SL(2)')
    parser.add_argument('--thresh', type=float, default=3, 
                        help='--Condition number maximum for generating SL2 matrices, when transform mode is on for some split')
    parser.add_argument('--cond_range', type=int, nargs="+", default=None)

    args = parser.parse_args()
    main(args)
