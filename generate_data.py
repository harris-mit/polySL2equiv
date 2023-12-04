# File generate_data.py
# Used to generate synthetic data

import numpy as np
from thewalrus import perm
import math
import cvxpy
from utils import sum_antidiagonals, batch_sum_antidiagonals
import torch
import argparse
import pickle
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
import sympy
from scipy.optimize import brute, fmin
from sympy.utilities.lambdify import lambdify
from collections import defaultdict
import torch
from cg import transvectant
from itertools import combinations
# seed random numbers so datasets are predictable!
import time
from tqdm import tqdm
import pickle

# to use to make slightly faster (collation of y's) if things work
def collate_tensor_fn(batch, *, collate_fn_map):
    return torch.stack(batch, 0)

# Was being used for pin_memory, but turning off pin_memory to debug and to avoid other unrelated errors from not returning a tuple
"""class CustomDictTensorBatch:
    def __init__(self, batch_x, batch_y):
        self.batch_x = batch_x # is a dictionary, where the values are already batched tensors
        self.batch_y = batch_y # is already a batched tensor
    
    def pin_memory(self):
        for ky in self.batch_x.keys():
            self.batch_x[ky] = self.batch_x[ky].pin_memory()
        self.batch_y = self.batch_y.pin_memory()
        return self"""

    

def collate_dicts(batch):
    # batch should be a list of tuples (x,y), where x is a dictionary and y is a scalar (torch.Tensor)
    # ASSUME the keys of x are ALWAYS the same 

    y_list = []
    x_list = []
    kys = batch[0][0].keys()
    batch_x = defaultdict(lambda: [])
    for elt in batch:
        x = elt[0]
        for ky in x.keys():
            batch_x[ky].append(x[ky])
        y_list.append(elt[1])
    for ky in kys:
        batch_x[ky] = torch.stack(batch_x[ky], dim=0)
    batch_y = torch.stack(y_list, dim=0)
    return batch_x, batch_y

class DictDatasetWithAug(Dataset):
    def __init__(self, x, y, transform=None, normalize=False):
        # x is a list of dictionaries with integer keys (equal to degree)
        # y is a tensor
        self.x = x
        self.y = y
        self.normalize = normalize
        self.aug = Augment(transform)
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x_orig = self.x[idx] # sizes [1], [2], [3], ...
        y_orig = self.y[idx] # size [] (just a number)

        # unsqueeze all
        x_unsqueezed = {}
        for ky in x_orig.keys():
            x_unsqueezed[ky] = x_orig[ky].unsqueeze(0) # now [1,1], [1,2], [1,3]
        y_unsqueezed = y_orig.unsqueeze(0).unsqueeze(0) # now [1, 1]
        
        if self.transform is not None:
            x_unsqueezed, y_unsqueezed = self.aug.apply_transform(x_unsqueezed, y_unsqueezed)
            
        if self.normalize:
            x_unsqueezed, y_unsqueezed = normalize_pair(x_unsqueezed, y_unsqueezed)

        # re-squeeze before returning
        for ky in x_unsqueezed.keys():
            x_unsqueezed[ky] = x_unsqueezed[ky].squeeze(0)
        y_unsqueezed = y_unsqueezed.squeeze(0).squeeze(0)

        return x_unsqueezed, y_unsqueezed
        #return self.x[idx], self.y[idx]


def induced(A, k, scaled = False, return_tensor=False):
    """
    Computes the induced matrix A^[k] 
    Assumes that this is for dimension n = 2
    Input matrix will be converted to numpy array

    Output is k+1 x k+1
    """
    n = 2
    num_terms = k + 1
    A = np.array(A)
    # would need below if n != 2
    # binom(k + n - 1, k)
    T = np.zeros((num_terms, num_terms))
    for i in range(num_terms):
        for j in range(num_terms):
            alpha = [k-i, i]
            beta = [k-j, j]
            idx1 = [q for q in range(len(alpha)) for r in range(alpha[q])]
            idx2 = [q for q in range(len(beta)) for r in range(beta[q])]
            repA = A[idx1, :][:, idx2]
            per = perm(repA)
            mualpha = np.prod(list(map(math.factorial, alpha)))
            mubeta = np.prod(list(map(math.factorial, beta)))
            T[i,j] = per
            if scaled:
                T[i,j] *= 1/(math.sqrt(mualpha) * math.sqrt(mubeta))
            else:
                T[i,j] *= 1/mubeta
    if return_tensor:
        return torch.tensor(T).float()
    else:
        return T


def get_random_poly_and_min(d):
    nummins = int(d/2)
    localminxy = torch.randn(nummins,2) # 4 local minima
    localbumps = torch.randn(nummins)
    minx, miny = localminxy.min(dim = 0).values
    maxx, maxy = localminxy.max(dim = 0).values
    x, y = sympy.symbols("x y")
    ppoly = 1
    for i in range(nummins):
        ppoly *= ((x - localminxy[i,0])**2 + (y-localminxy[i,1])**2 + localbumps[i]**2)
    p = sympy2homogdict(ppoly) # is a dictionary
    optim_results = optim2dsympy(ppoly, (minx.item(), maxx.item()), (miny.item(), maxy.item()))
    return p, optim_results[1], optim_results[0] # the minimum value

def get_random_poly_and_min_structured(d):
    # local minima are on a grid
    d = int(d/2); # deg = 2 * d
    minval = np.random.randn()
    xval, yval = np.abs(np.random.randn(2))
    gxval, gyval = np.multiply(np.random.rand(2),np.array([2*xval, 2*yval])) - np.array([xval, yval])
    localminxy = np.array([[-xval, -yval],
                           [-xval, yval],
                           [gxval, gyval],
                           [xval, -yval],
                           [xval, yval]])
    x, y = sympy.symbols("x y")
    p = minval
    indexlist = list(combinations(range(localminxy.shape[0]), d-1))
    for i in range(len(indexlist)):
        ptemp = 1
        for j in indexlist[i]:
            ptemp *= ((x - localminxy[j,0])**2 + (y - localminxy[j,1])**2)
        p += ptemp * ((x - gxval)**2 + (y - gyval)**2)
    return sympy2homogdict(p), np.array([gxval,gyval]), minval
    
    
def optim2dsympy(ppoly, xbounds, ybounds):
    # (idea from https://stackoverflow.com/questions/34115233/python-optimization-using-sympy-lambdify-and-scipy)
    x, y = sympy.symbols("x y")
    my_ppoly = lambdify((x,y), ppoly)
    if xbounds[0] == xbounds[1] and ybounds[0] == ybounds[1]: # just a single local min
        return [torch.tensor([xbounds[0], ybounds[0]]), my_ppoly(xbounds[0], ybounds[0])]
    rranges = (slice(xbounds[0], xbounds[1], .25), slice(ybounds[0], ybounds[1], .25))
    def my_ppoly_func_v(z):
        return my_ppoly(z[0], z[1])
    results = brute(my_ppoly_func_v, rranges, full_output=True,
                     finish = fmin)
    return results

def sympy2homogdict(ppoly):
    p = sympy.Poly(ppoly)
    termdegs = list(map(sum,p.as_dict().keys()))
    p_dicts = {}
    for ky in termdegs: # all of the degrees
        binary_form = torch.zeros(ky + 1)
        for xi in range(ky+1):
            try:
                binary_form[xi] = float(p.as_dict()[xi, ky-xi])
            except KeyError: # not in p
                binary_form[xi] = 0
            except:
                print("error other than key")
            p_dicts[ky] = binary_form
    return p_dicts

def get_rotations(nummatrices, return_tensor=False):
    lst = []
    for i in range(nummatrices):
        theta = np.random.uniform(0,2*np.pi)
        q = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), (np.cos(theta))]])
        lst.append(torch.tensor(q))
    allmats = torch.stack(lst, dim=0)
    if not return_tensor:
        return np.array(allmats)
    else:
        return allmats

def sample_gaussian_SL2():
    mat = np.random.randn(2,2)
    det = np.linalg.det(mat) 
    if det < 0:
        mat[0,:] *= -1
        det = abs(det)
    mat = torch.tensor( mat / np.sqrt(det))
    return mat

def sample_non_gaussian_SL2():
    theta = np.random.uniform(0, 2*np.pi)
    r = (np.random.randn()+1)**2
    x = np.random.randn()
    K = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = np.array([[r, 0], [0, 1/r]])
    N = np.array([[1, x], [0, 1]])
    mat = np.matmul(np.matmul(K, A), N)
    mat = torch.tensor(mat)

def return_range_ind(condnum, cond_range):
    # TODO: don't include ones that are below the first value in cond_range
    # assume cond_range is sorted in increasing order

    for (i, num) in enumerate(cond_range):
        # return -1 if condnum < cond_range[0] or condnum > cond_range[-1]
        if condnum < num:
            if i == 0:
                return -1
            else:
                return i - 1
    return -1 # return -1 if invalid
def get_random_SL2(mode, diag=False):
    if mode == 'gaussian':
        mat = sample_gaussian_SL2()
    else: 
        mat = sample_non_gaussian_SL2()
    return mat

def get_SL2_matrices(nummatrices, return_tensor=False, mode='gaussian', thresh=3, cond_lb=None, cond_range=None, diag=False):
    # mode is gaussian or iwasawa
    if diag and nummatrices == 1: # hacky 

        assert cond_range == None
        cond_num = np.random.uniform(cond_lb, thresh) 
        mat = torch.zeros((1, 2, 2))
        mat[0] = torch.tensor([[np.sqrt(cond_num), 0], [0, 1.0 / np.sqrt(cond_num)]])
    if cond_range is None:
        lst = []
        if cond_lb is None:
            cond_lb = -1
        for i in range(nummatrices): # need to change this 
            firsttry = True
            while firsttry or condnum > thresh or condnum < cond_lb:
                mat = get_random_SL2(mode)
                if firsttry:
                    firsttry = False
                condnum = np.linalg.cond(mat)
            lst.append(mat) 
    else:
        cond_nums = []
        cond_ids = []
        counts = np.zeros(len(cond_range)-1)
        lst = []
        while np.min(counts) < nummatrices: # nummatrices is interpreted as PER count
            firsttry = True
            while firsttry or cond_range_ind < 0:
                mat = get_random_SL2(mode)
                condnum = np.linalg.cond(mat)
                cond_range_ind = return_range_ind(condnum, cond_range)
                print(f'condnum {condnum} cond_range_ind {cond_range_ind}')
                if firsttry:
                    firsttry = False
            counts[cond_range_ind] += 1 # ?? only if not first
            lst.append(mat)
            cond_ids.append(cond_range_ind)
            cond_nums.append(condnum)
        if return_tensor:
            cond_nums = torch.tensor(cond_nums)
            cond_ids = torch.tensor(cond_ids)
    allmats = torch.stack(lst, dim=0)
    if not return_tensor:
        allmats = np.array(allmats)
    if cond_range is None:
        return allmats
    else:
        return allmats, cond_nums, cond_ids

def transform_poly_dict(poly, A, A_induceds_input=None):
    newpoly = {}
    if A_induceds_input is not None and type(A_induceds_input) == list:
        dict_version = {}
        for i in range(len(A_induceds_input)):
            dict_version[f'{i}'] = A_induceds_input[i]
        A_induceds_input = dict_version
    A_induceds = {}
    for ky in poly.keys():
        deg = ky
        if A_induceds_input is None:
            A_induced = induced(A, k=deg, scaled=False, return_tensor=True).to(poly[ky].device)
        else:
            A_induced = A_induceds_input[f'{deg}']
        A_induceds[f'{deg}'] = A_induced
        newpoly[ky] = torch.matmul(A_induced.permute(1,0), poly[ky].unsqueeze(-1))
        if len(newpoly[ky].shape) == 3:
            newpoly[ky] = newpoly[ky][:, :, 0] # deg+1 x 1
    return newpoly, A_induceds

def transform_many_polys(polys, As): # MIGHT HAVE DIMENSION ERROR? 
    # mats is batch x deg          probably a tensor
    # As is   batch x 2    x  2    numpy array (or will be converted)
    # transform each element of polys by appropriately induced matrix of each element of As
    if type(polys) == type(np.random.rand(3)):
        polys = torch.tensor(polys)
    if type(As) != torch.Tensor:
        As = torch.tensor(As)
    transformed_polys = torch.zeros(polys.shape)

    batch = polys.shape[0]
    sz = polys.shape[1]
    deg = sz - 1

    for i in range(batch):
        A_induced = induced(As[i], k=deg, scaled=False, return_tensor=True) #deg+1 x deg + 1
        x = polys[i] # one dimension (deg+1)
        res = torch.matmul(A_induced.permute(1,0), x.unsqueeze(-1)) # deg+1 x 1
        transformed_polys[i, :] = res[:,0]
        
    return transformed_polys

def get_many_random_pos_poly(numpolys, d):
    """
    Generates many random polynomials that are definitely nonnegative

    d: 2*d is degree of polynomials
    """
    A = torch.randn(numpolys, d+1, d+1)
    return batch_sum_antidiagonals(torch.bmm(A.permute(0, 2, 1), A) + 1e-8 * torch.eye(d+1).repeat(numpolys, 1, 1))
# add 10^-8 so that the polynomial is strictly positive

def get_random_pos_poly(d):
    """
    Generates a random polynomial that is definitely nonnegative

    d: 2*d is degree of polynomial
    """
    A = torch.randn(d+1, d+1)
    return sum_antidiagonals(torch.transpose(A, 0, 1)@A + 1e-8 * torch.eye(d+1))

def get_max_log_det(p, solver = "MOSEK"):
    """
    Finds the max log determinant Gram matrix with cvxpy.
    """
    d = int((p.shape[0] - 1)/2)
    if True:
        solver = 'SCS'
    X = cvxpy.Variable((d+1, d+1), symmetric = True)
    constraints = []
    for monomial_degree in range(2*d+1): # match monomials
        this_term = 0
        for i in range(max(0,monomial_degree - d), min(monomial_degree,d)+1):
            j = monomial_degree - i
            this_term += X[i, j]
        constraints += [this_term == p[monomial_degree]]
    start = time.time() 
    prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.atoms.log_det(X)), constraints)
    #print('should be verbose')
    prob.solve(solver = solver, verbose=False)
    end= time.time()
    #print('X', type(X), X)
    #print('value', X.value)
    #print('Degree', p.shape[0] - 1, 'time for ${solver} to solve 1 instance', (end - start)/60, 'seconds')
    return torch.tensor(X.value)

def batch_ppn(ps):
    """
    Given ps is batch x size, compute psi(ps[i], ps[i], size-1) for all i
    """
    tns = torch.zeros(ps.shape[0])
    n = ps.shape[1] - 1
    for i in range(len(tns)):
        tns[i] = transvectant(ps[i], ps[i], n)
    return tns

class Augment():
    def __init__(self, transform):
        if type(transform) == dict:
            # load matrices and induced matrices
            self.presaved = True
            transform_file = transform['transform_file']
            with open(transform_file, 'rb') as f:
                out = pickle.load(f)
            self.As = out['As']
            self.induced = out['induced']
            self.num_presaved_augs = len(self.As)
            self.transform_from_presaved_fxn = transform['transform_from_presaved_fxn'] 
        else: 
            self.presaved = False
            self.transform = transform
    
    def apply_transform(self, x, y):
        if self.presaved:
            ind = np.random.randint(self.num_presaved_augs)
            return self.transform_from_presaved_fxn(x=x, y=y, A=self.As[ind], induced_mats=self.induced[ind])
        else:
            return self.transform(x, y)

class TensorDatasetWithAug(Dataset):
    def __init__(self, tensors, transform=None, normalize=False):
        # transform is None for no transform
        # transform is fxn that takes in x,y for non-precomputed version
        # transform is dictionary with 'transform_file' key for precomputed file full path and 'transform_from_presaved_fxn'
        # key that takes as input x, y, A, induced_mats
        super().__init__()
        self.tensors = tensors
        self.aug = Augment(transform)
        self.transform = transform
        self.normalize = normalize
    
    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform is not None:
            x, y = self.aug.apply_transform(x.unsqueeze(0), y.unsqueeze(0))
        
        if self.normalize:
            x, y = normalize_pair(x, y)

        return x.squeeze(0), y.squeeze(0)

    def __len__(self):
        return self.tensors[0].size(0)

def normalize_pair(x, y):
    # MAX_DET:
    # x is batch x deg+1
    # y is batch x deg/2 + 1 x (deg/2 + 1)
    # 
    # MIN POLY:
    # x is a list of dictionaries with integer keys (equal to degree). is it homogeneous? 
    # y is a tensor
    if type(x) == torch.Tensor:
        facs = torch.norm(x, dim=-1).reshape(-1, 1) # size: batch x 1
        normalized_x = x / facs 
        normalized_y = y / facs.reshape(-1, 1, 1)
    else:
        # normalize by norm of entire input

        x_size_changed = False
        x_orig_sizes = {}
        for ky in x.keys():
            x_orig_sizes[ky] = x[ky].shape
        for ky in x.keys():
            if len(x[ky].shape) < 2:
                assert False, 'given x[ky] of shape above in normalize_pair'
                x_size_changed = True
                x[ky] = x[ky].unsqueeze(0)
        orig_y_shape = y.shape
        if len(y.shape) < 2:
            y = y.reshape(y.shape[0], 1)
            
        num_batch = y.shape[0]
        facs = torch.zeros((num_batch, 1), device=y.device)
        for ky in x.keys(): 
            facs += (torch.norm(x[ky], dim=-1)).reshape(-1, 1) # size: batch x 1

        facs = torch.sqrt(facs)

        normalized_x = {}
        for ky in x.keys():
            normalized_x[ky] = x[ky] / facs
            normalized_x[ky].reshape(x_orig_sizes[ky])
            
        normalized_y = y / (facs)
        normalized_y = normalized_y.reshape(orig_y_shape )
            
        #assert num_keys == 1, "Cannot scale non-homogeneous polynomial for minimization problem"
    return normalized_x, normalized_y

def pickle_to_dataloader(data_dir, batch_size, return_datasets=False, transvectant_scaling=False, transform=None, normalize=False):
    dsets = {}
    dloaders = {}
    doshuffle = {'train': True, 'val': False, 'test': False} 
    for split in ['train', 'val', 'test']:
        if split != 'train':
            transform = None # do not transform the validation or test set 
        with open(os.path.join(data_dir, f'{split}.pkl'), 'rb') as f:
            out = pickle.load(f)
        x = out['x'] # batch x deg+1 for max_det
        y = out['y'] # batch x (deg/2 + 1) x (deg/2 + 1) for max_det
        if transvectant_scaling:
            if type(x) == torch.Tensor: # Assuming this is max_det
                bppn_out = batch_ppn(x)
                scales = torch.sqrt(torch.abs(bppn_out))
                scales = torch.maximum(scales, torch.ones(scales.shape, device=scales.device))
                x = torch.div(x.permute(1,0), scales).permute(1,0)
                avg_norm_after = torch.norm(x)/x.shape[0]
                x = x / avg_norm_after
                y = torch.div(y, scales.unsqueeze(-1).unsqueeze(-1))
            else: # min_poly
                print("Scaling not implemented for min_poly")
        if type(x) == torch.Tensor:
            dsets[split] = TensorDatasetWithAug(tensors=(x, y), transform=transform, normalize=normalize)
            dloaders[split] = DataLoader(dsets[split], batch_size=batch_size, num_workers=4, shuffle=doshuffle[split])
        else:
            dsets[split] = DictDatasetWithAug(x, y, normalize=normalize, transform=transform)
            dloaders[split] = DataLoader(dsets[split], batch_size=batch_size, num_workers=0, shuffle=doshuffle[split], pin_memory=torch.cuda.is_available()) #collate_fn=collate_dicts, )
        
    if return_datasets:
        return dsets, dloaders
    else:
        return dloaders

def main(args: argparse) -> None:

    torch.manual_seed(0)
    np.random.seed(0)
    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.expanduser('~'), 'TransvectantNets_shared')
        if args.mode == 'max_det':
            args.data_dir = os.path.join(args.data_dir, 'data/equivariant/')
        elif args.mode == 'min_poly':
            args.data_dir = os.path.join(args.data_dir, 'data/min_poly/')
        else:
            args.data_dir = os.path.join(args.data_dir, 'data/invariant/')
        args.data_dir += f'deg_{args.input_degree}_train_{args.num_train}_val_{args.num_val}_test_{args.num_test}'
        for split, do_transform in zip(['train', 'val', 'test'], [args.transform_train, args.transform_val, args.transform_test]):
            if do_transform:
                args.data_dir += f'_transform_{split}'
        if do_transform:
            args.data_dir += f'_{args.transform_mode}_{args.thresh}'

    thresh = args.thresh #np.sqrt(3)
    print('Saving data in directory', args.data_dir)
    data_dir = os.makedirs(args.data_dir, exist_ok=True)
    deg = args.input_degree
    for split, numsamples, do_transform in zip(['train', 'val', 'test'], [args.num_train, args.num_val, args.num_test], [args.transform_train, args.transform_val, args.transform_test]):
        x = []
        if args.mode == 'max_det':
            x = get_many_random_pos_poly(numsamples, int(deg / 2))
        elif args.mode == 'def':
            x = get_many_random_pos_poly(np.floor(numsamples / 2), int(deg / 2))
            x = torch.cat((torch.randn(numsamples - np.floor(numsamples / 2), int(deg / 2) + 1), x), 0)
        else:
            print("Generating data for minimizing polynomials...")

        if do_transform and (args.mode == 'max_det' or args.mode == 'def'):
            if args.transform_mode == 'rotations':
                As = get_rotations(numsamples, return_tensor=True)
            else:
                As = get_SL2_matrices(numsamples, return_tensor=True, mode=args.transform_mode, thresh=thresh)
            x = transform_many_polys(x, As)

        y = []
        for i in tqdm(range(numsamples)):
            if args.mode == 'max_det':
                mat = get_max_log_det(x[i, :]).float()
                y.append(mat) 
            elif args.mode == 'def':
                res = torch.linalg.eigh(get_max_log_det(x[i,:]))
                res2 = torch.linalg.eigh(get_max_log_det(-1*x[i,:]))
                if (res.eigenvalues>0).all() or (res2.eigenvalues>0).all():
                    is_definite = 1
                else:
                    is_definite = 0
                y.append(is_definite) # requires classification loss
            elif args.mode == 'min_poly' or args.mode == 'minimizer_poly':
                if args.structured_poly_min:
                    p, theminimizer, themin = get_random_poly_and_min_structured(deg)
                else:
                    p, themin, theminimizer = get_random_poly_and_min(deg)
                if do_transform:
                    singleA = get_SL2_matrices(1, return_tensor=True, mode=args.transform_mode, thresh=thresh)
                    p = transform_poly_dict(p, singleA[0])
                x.append(p)
                # don't we need to torch.stack the x's as well? perhaps this is done automatically in pytorch dataset/dataloader
                if args.mode == 'min_poly':
                    y.append(torch.tensor(themin))
                else:
                    # TODO: TRANSFORM BY singleA
                    y.append(torch.tensor(theminimizer)) # TODO: check that this works with custom dataset
            else:
                print("Invalid mode")
        y = torch.stack(y, dim=0)

        with open(os.path.join(args.data_dir, f'{split}.pkl'), 'wb') as f:
            pickle.dump({'x': x, 'y': y}, f) 

        # TEST WHETHER CAN LOAD DATALOADERS AND DATASETS
    dsets, dloaders = pickle_to_dataloader(args.data_dir, 10, return_datasets=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tuning an MAE model')
    #parser.add_argument('--chkpt', type=str, default='checkpoints/temp.pt',
                        #help='Path to save checkpoint')
    parser.add_argument('--num_train', type=int, default=10,
                        help='Number of training instances to generate')
    parser.add_argument('--num_val', type=int, default=10,
                        help='Number of validation instances to generate')
    parser.add_argument('--num_test', type=int, default=10,
                        help='Number of test instances to generate')

    parser.add_argument('--transform_train', default=False, action='store_true',
                        help='Transform elements of training set. NOT yet implemented for minimizer_poly')
    parser.add_argument('--transform_val', default=False, action='store_true',
                        help='Transform elements of val set. NOT yet implemented for minimizer_poly')
    parser.add_argument('--transform_test', default=False, action='store_true',
                        help='Transform elements of test set. NOT yet implemented for minimizer_poly')

    parser.add_argument('--transform_mode', type=str, default='gaussian',
                        help='gaussian or iwasawa as distribution over SL(2)')
                

    parser.add_argument('--input_degree', type=int, default=8,
                        help='--Input degree of polynomials')
    parser.add_argument('--structured_poly_min', default=False, action='store_true',
                        help='Use local minima on a grid to construct minimizer polynomials')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='--Path to directory to save files as train.pkl, test.pkl, val.pkl. Will create directory if it doesn\'t exist already')
    parser.add_argument('--mode', type=str, default='max_det',
                        help='--Mode for generating labels y. --max_det for Gram matrix of maximal determinant (equivariant, matrix task), --def for definiteness (invariant, scalar task), --min_poly for minimizing a polynomial, --minimizer_poly for getting the location of the minimizer of a polynomial.')

    parser.add_argument('--thresh', type=float, default=1.1, 
                        help='--Condition number maximum for generating SL2 matrices, when transform mode is on for some split')
                        
    args = parser.parse_args()
    main(args)

