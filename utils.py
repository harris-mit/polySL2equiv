# File: utils.py
# Includes all the the utility functions for the project!

import torch
from torch import nn
from tests.bernstein_comparison import ispsd #to put back

# A polynomial is represented as coefficients of
# [1, x, x^2, x^3, ... x^d]

def batch_multiply_poly_tensors(x: torch.Tensor, y: torch.Tensor):
    # Inputs are: x = (batch, m)
    #             y = (batch, n)
    # Output:         (batch, m+n-1)

    z = torch.bmm(x.unsqueeze(-1), y.unsqueeze(1))
    return batch_sum_antidiagonals(z)

def batch_sum_antidiagonals(z: torch.Tensor):
    # z is a 3D tensor, z = (batch, n1, n2)
    # output is 2D,         (batch, n1 + n2 - 1)
    # and n1 
    b, n1, n2 = z.shape
    zpad = torch.cat((z, torch.zeros((b, n1, n1 - 1), device=z.device)), -1)
    zpad = zpad.as_strided(zpad.shape, (zpad.shape[2]*n1, zpad.shape[2]-1,1))
    return torch.sum(zpad, 1) # sums the columns


def multiply_poly_tensors(x: torch.Tensor, y: torch.Tensor):

    """
    Assumes that x and y are single dimensional arrays whose entries represent
    the coefficients in the monomial basis
    [y^d, x*y^(d-1), ..., x^d] (here x and y are not input but the variables x,y)
    So [0, 1, 0, 0] represents the polynomial xy^2
    And [2, 0, 0, 1, -1] represents 2y^4 + x^3y - x^4
    >>> a = torch.Tensor([0,1,0,0])
    >>> b = torch.Tensor([2, 0, 0, 1, -1])
    >>> multiply_poly_tensors(a, a)
    tensor([0., 0., 1., 0., 0., 0., 0.])

    We get x^2y^4 as expected.
    >>> multiply_poly_tensors(a, b)
    tensor([ 0.,  2.,  0.,  0.,  1., -1.,  0.,  0.])

    a*b is 2xy^6 + x^4 y^3 - x^5y^2

    If x and y are lengths m and n, then the output size is m + n - 1

    Credit for the implementation idea from 
    https://stackoverflow.com/questions/57347896/sum-all-diagonals-in-feature-maps-in-parallel-in-pytorch
    """
    z = torch.outer(x, y)
    return sum_antidiagonals(z)

def sum_antidiagonals(z: torch.Tensor):
    # z is a 2D tensor

    n1, n2 = z.shape
    zpad = torch.cat((z, torch.zeros((n1, n1 - 1), device=z.device)), 1)
    zpad = zpad.as_strided(zpad.shape, (zpad.shape[1]-1,1))
    return torch.sum(zpad, 0) # sums the columns

def count_parameters(model): 
    # From https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def diff_normalized_mse_loss(x, y, normalizer):
    if x.shape != y.shape:
        print('ERROR: in normalized_mse_loss, shapes are', x.shape,'and ', y.shape)
        return
    if x.shape[0] != y.shape[0] or len(x.shape) < 2:
        #print('ERROR: normalized_mse_loss expects a batch dimension, but shapes are', x.shape,'and ', y.shape)
        #return
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        normalizer = normalizer.unsqueeze(-1)
    if len(x.shape) == 3:
        dims_to_reduce = (1,2)
    else:
        if len(x.shape) == 2:
            dims_to_reduce = (1)
    normalized_loss = torch.mean(torch.divide(torch.sum(torch.square(x - y),dim=dims_to_reduce), torch.sum(torch.square(normalizer),dim=dims_to_reduce) + 1))
    #normalized_loss = torch.divide(torch.square(torch.norm(x - y)), torch.square(torch.norm(y)) + 1)
    return normalized_loss

def normalized_mse_loss(x, y):
    if x.shape != y.shape:
        print('ERROR: in normalized_mse_loss, shapes are', x.shape,'and ', y.shape)
        return
    if x.shape[0] != y.shape[0] or len(x.shape) < 2:
        #print('ERROR: normalized_mse_loss expects a batch dimension, but shapes are', x.shape,'and ', y.shape)
        #return
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
    if len(x.shape) == 3:
        dims_to_reduce = (1,2)
    else:
        if len(x.shape) == 2:
            dims_to_reduce = (1)
    normalized_loss = torch.mean(torch.divide(torch.sum(torch.square(x - y),dim=dims_to_reduce), torch.sum(torch.square(y),dim=dims_to_reduce) + 1))
    #normalized_loss = torch.divide(torch.square(torch.norm(x - y)), torch.square(torch.norm(y)) + 1)
    return normalized_loss

def prepare_for_logger(kwargs):
    mydict = {}
    badkeys = []
    for ky in kwargs.keys():
        if type(kwargs[ky]) == type([1,2]):
            for ind, elt in enumerate(kwargs[ky]):
                mydict[f'ky_{ind}'] = elt
        elif type(kwargs[ky]) == type("abc"):
            mydict[kwargs[ky]] = 1
        else:
            mydict[ky] = kwargs[ky]
    return mydict
        


if __name__ == "__main__":
    import doctest
    doctest.testmod()

def fraction_psd(mats, cutoff=0.0):
    # mats is batch x dim x dim
    numpsd = 0.0
    numbatch = mats.shape[0]
    with torch.no_grad():
        for i in range(numbatch):
            if ispsd(mats[i], cutoff=cutoff):
                numpsd += 1
    return numpsd / numbatch
