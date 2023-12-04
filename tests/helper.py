# Helpers that are useful in potentially many tests

import numpy as np
#from thewalrus import perm
import math
import torch
import sys
sys.path.append('..')
from generate_data import get_many_random_pos_poly, get_SL2_matrices, induced, transform_poly_dict, get_rotations
# transform_poly_dict(poly, A) 

def get_SL2_and_induceds(deg, thresh, cond_lb, diag=False, min_poly=False):
    # single SL2 matrix and induced matrices

    if cond_lb == 1 and thresh == 1:
        A = get_rotations(1, return_tensor=False)[0, ...]
    else:
        A = get_SL2_matrices(1, return_tensor=False, thresh=thresh, cond_lb=cond_lb, diag=diag)[0, ...]
    #deg = x.shape[-1] - 1
    if min_poly:
        return A
        """induced_mats = []
        for k in range(args.max_degree + 1): # max degree or something else? no I think degree is right 
            induced_deg_k = induced(A, k=k, scaled=False, return_tensor=True)
            induced_mats.append(induced_deg_k)
        return A, induced_mats"""
    else:
        n = int((deg / 2) + 1)
        if cond_lb == 1 and thresh == 1:
            A = get_rotations(1, return_tensor=False)[0, ...]
        else:
            A = get_SL2_matrices(1, return_tensor=False, thresh=thresh, cond_lb=cond_lb)[0, ...]
        A_induced = induced(A, k=deg, scaled=False, return_tensor=True) #.to(x.device)
        A_out_induced = induced(A, k=n-1, scaled=False, return_tensor=True) #.to(x.device)
        return A, A_induced, A_out_induced

def equiv_transform(x, pred, thresh=3, cond_lb=None, rotation=False, return_A=False, A_and_induceds = None, mode='max_det'):
    # eventually: do different A for each batch element
    # eventually: call this function in tests/equivariance.py 
    # x is batch x deg+1
    # min_poly: A_and_induceds is just A | matches get_SL2_and_induceds
    # max_det: A_and_induceds is A, A_induced, A_out_induced | matches get_SL2_and_induceds
    # 
    #print('IN EQUIV_TRANSFORM', 'A_and_induceds', A_and_induceds)
    if A_and_induceds is None:
        if rotation: 
            A = get_rotations(1, return_tensor=False)[0, ...]
        else:
            A = get_SL2_matrices(1, return_tensor=False, thresh=thresh, cond_lb=cond_lb)[0, ...]
        if mode == 'max_det':
            deg = x.shape[-1] - 1
            n = int((deg / 2) + 1)
            A_induced = induced(A, k=deg, scaled=False, return_tensor=True).to(x.device)
            A_out_induced = induced(A, k=n-1, scaled=False, return_tensor=True).to(x.device)
            return_A_and_induceds = {'A': A, 'A_induced': A_induced, 'A_out_induced': A_out_induced}
        else:
            return_A_and_induceds = {'A': A}
    else:
        if mode == 'max_det':
            A, A_induced, A_out_induced = A_and_induceds
            return_A_and_induceds = {'A': A, 'A_induced': A_induced, 'A_out_induced': A_out_induced}
        else:
            A = A_and_induceds
            return_A_and_induceds = {'A': A}

    if mode == 'max_det':
        x_transformed = torch.matmul(A_induced.permute(1,0), x.unsqueeze(-1))[:,:,0]
        pred_transformed = torch.matmul(torch.matmul(A_out_induced.permute(1, 0), pred), A_out_induced)
    else:
        x_transformed, A_induceds = transform_poly_dict(x, A)
        pred_transformed = pred
        return_A_and_induceds['induceds'] = A_induceds


    if return_A:
        return x_transformed, pred_transformed, return_A_and_induceds
    else:
        return x_transformed, pred_transformed

def equiv_transform_for_min_poly(x, pred, thresh=3, cond_lb=None, rotation=False, return_A=False, A_and_induceds = None):
    return equiv_transform(x, pred, thresh=thresh, cond_lb=cond_lb, rotation=rotation, return_A=return_A, A_and_induceds = A_and_induceds, mode='min_poly')


# NOT USED ANYMORE
"""def equiv_transform_for_min_poly(x, pred, thresh=3, cond_lb=None, rotation=False):
    # x is a dictionary
    if rotation:
        A = get_rotations(1, return_tensor=False)[0, ...]
    else:
        A = get_SL2_matrices(1, return_tensor=False, thresh=thresh, cond_lb=cond_lb)[0, ...]
    transformed_x = transform_poly_dict(x, A)
    return transformed_x, pred"""

def equiv_transform_for_min_poly_from_presaved_fxn(x, y, A, induced_mats):
    #print('induced_mats in equiv_transform_for_min_poly_from_presaved_fxn', type(induced_mats))
    x_transformed, A_induceds = transform_poly_dict(x, A, A_induceds_input=induced_mats)
    return x_transformed, y


def equiv_transform_from_presaved_fxn(x, y, A, induced_mats):
    # A is 2x2 matrix
    # induced_mats goes from k=0 up to suitably high (hopefully) max
    # (pred is renamed y)
    deg = x.shape[-1] - 1
    n = int((deg / 2) + 1)
    A_induced = induced_mats[deg] #induced(A, k=deg, scaled=False, return_tensor=True).to(x.device)
    A_out_induced = induced_mats[n-1] #induced(A, k=n-1, scaled=False, return_tensor=True).to(x.device)
    
    x_transformed = torch.matmul(A_induced.permute(1,0), x.unsqueeze(-1))[:,:,0]
    pred_transformed = torch.matmul(torch.matmul(A_out_induced.permute(1, 0), y), A_out_induced)
    return x_transformed, pred_transformed


        
