# File: cg.py
# Computes the CG transform for the tensor product for
# pointwise polynomial multiplication

import math
import torch
import utils
from scipy.special import binom

def differentiate_poly(p, v, order = 1):
    """
    Differentiates the given polynomial (input in the usual tensor form)
    Arguments:
    p: torch.Tensor representing polynomial
    v: "0" for x and "1" for y
    order: differentiation order
    This uses recursion and is not intended for efficiency...

    We differentiate x^4 * y^2 + 2.2 * x^3 * y^3 - y^6
    >>> p = torch.Tensor([-1, 0, 0, 2.2, 1, 0, 0])
    >>> differentiate_poly(p, 0, 1) # d/dx(p)
    tensor([0.0000, 0.0000, 6.6000, 4.0000, 0.0000, 0.0000])
    >>> differentiate_poly(p, 0, 2) # d^2/dx^2 (p)
    tensor([ 0.0000, 13.2000, 12.0000,  0.0000,  0.0000])
    >>> differentiate_poly(p, 1, 1) # d/dy (p)
    tensor([-6.0000,  0.0000,  0.0000,  6.6000,  2.0000,  0.0000])
    >>> differentiate_poly(p, 1, 4) # d^4/dy^4 (p)
    tensor([-360.,    0.,    0.])
    """
    if order == 0:
        return p
    if order == 1:
        idxvec = torch.arange(p.shape[0])
        if v == 0:
            return torch.multiply(idxvec, p)[1:]
        else: # v == 1
            idxvec = torch.flip(idxvec, (0,))
            return torch.multiply(idxvec, p)[:-1]
    else:
        return differentiate_poly(differentiate_poly(p, v, 1), v, order - 1)


def transvectant(f, g, n, numerically_stable = True):
    """
    Computes the nth order transvectant of f and g
    These computations were checked against my Julia implementation.
    Let f = x^2 * y + y^3, g = x^2 - 2*y^2
    >>> f = torch.Tensor([1, 0, 1, 0])
    >>> g = torch.Tensor([-2, 0,1])
    >>> transvectant(f, g, 0)
    tensor([-2.,  0., -1.,  0.,  1.,  0.])
    >>> transvectant(f, g, 1)
    tensor([ 0.0000, -2.3333,  0.0000, -0.3333])
    >>> transvectant(f, g, 2)
    tensor([0.3333, 0.0000])

    Transvectant is antisymmetric for odd n and symmetric for even
    >>> transvectant(f, g, 1) + transvectant(g, f, 1) 
    tensor([0., 0., 0., 0.])
    >>> transvectant(f, g, 2) + transvectant(g, f, 2) - 2 * transvectant(f, g, 2)
    tensor([0., 0.])
    """
    if n == 0:
        return utils.multiply_poly_tensors(f, g)
    d1 = f.shape[0] - 1
    d2 = g.shape[0] - 1
    psi = 0
    factor = math.factorial(d1 - n) / math.factorial(d1) * math.factorial(d2-n) / math.factorial(d2)
    if numerically_stable:
        rtfactor = math.sqrt(factor) # so we can scale each derivative...
        for i in range(n+1):
            deriv1 = differentiate_poly(differentiate_poly(
                f, 1, i), 0, n-i) * rtfactor
            deriv2 = differentiate_poly(differentiate_poly(
                g, 1, n-i), 0, i) * rtfactor
            binom_factor = math.comb(n,i)
            psi += ((-1)**i * binom_factor * utils.multiply_poly_tensors(deriv1, deriv2))
    else:
        for i in range(n+1):
            deriv1 = differentiate_poly(differentiate_poly(
                f, 1, i), 0, n-i)
            deriv2 = differentiate_poly(differentiate_poly(
                g, 1, n-i), 0, i)
            binom_factor = math.comb(n,i)
            psi += (factor * (-1)**i * binom_factor * utils.multiply_poly_tensors(deriv1, deriv2))
    return psi


def generate_linear_map(m, n, numerically_stable = True, row_normalize=False):
    """
    Given the shape of the input tensor, generate the map going from
    flattened tensor (x.reshape(-1,1)) to irreps
    Apply the resulting matrix to
    torch.outer(f,g).reshape(-1,1)
    
    >>> generate_linear_map(2,2)
    tensor([[ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  1.,  0.],
            [ 0.,  0.,  0.,  1.],
            [ 0., -1.,  1.,  0.]])

    The first column is the image of y tensor y
    psi_0(y,y) = y^2, and psi_1(y,y) = 0
    The second column is the image of y tensor x
    psi_0(y,x) = x*y and psi_1(y,x) = -1
    The difference in the third column is that
    psi_1(x,y) = 1 by anti-symmetry
    The fourth column is x^2 + 0
    """
    T = torch.zeros(m*n, m*n)
    tnsizes, tnorders = get_irrep_dims(m, n)
    for i in range(m):
        for j in range(n):
            thisf = torch.zeros(m)
            thisg = torch.zeros(n)
            thisf[i] = 1.
            thisg[j] = 1.
            for tni in range(len(tnsizes)):
                this_poly_idx = slice(sum(tnsizes[0:tni]),sum(tnsizes[0:tni+1]))
                T[this_poly_idx, j + i * n] = transvectant(thisf, thisg, tnorders[tni], numerically_stable) 
    # this will mess up the last layer for max det problem! 
    if row_normalize:
        T = T / torch.norm(T, dim=1).reshape(-1, 1)
    return T

# THIS FUNCTION IS UNUSED AND UNTESTED
def generate_symmetric_linear_mat(m, n):
    """
    Assuming the input tensors are symmetric, compute the map
    from tensor space to irreps (now only even irreps appear)
    """
    assert m==n, "If they're not equal, the tensor can't be symmetric"
    tri_size = int(n * (n + 1) / 2)
    T = torch.zeros(tri_size, tri_size)
    general_tnsizes, general_tnorders = get_irrep_dims(m, n)
    sym_tnorders = [x for x in general_tnorders if x % 2 == 0]
    sym_tnsizes = [general_tnsizes[i] for i in range(len(general_tnsizes)) if general_tnorders[i] % 2 == 0]
    for i in range(m):
        for j in range(i+1): # 0 to i...
            thisf = torch.zeros(m)
            thisg = torch.zeros(n)
            thisf[i] = 1
            thisg[j] = 1
            for tni in range(len(sym_tnsizes)):
                this_poly_idx = slice(sum(sym_tnsizes[0:tni]),sum(sym_tnsizes[0:tni+1]))
                if i == j:
                    T[this_poly_idx, j + i**2] = transvectant(thisf, thisg, sym_tnorders[tni])
                else:
                    T[this_poly_idx, j + i**2] = (transvectant(thisf, thisg, sym_tnorders[tni]) +
                                                 transvectant(thisg, thisf, sym_tnorders[tni]))
                    # If we see something off the main diagonal, should assume it appears twice (on other side too...)
    return T
# THIS FUNCTION IS UNUSED AND UNTESTED
def apply_inverse_symmetric_isomorphism(m, n, T, x):
    T = generate_symmetric_linear_mat(m, n)
    general_tnsizes, general_tnorders = get_irrep_dims(m, n)
    sym_tnorders = [x for x in general_tnorders if x % 2 == 0]
    sym_tn_degs = [general_tnsizes[i] - 1 for i in range(len(general_tnsizes)) if general_tnorders[i] % 2 == 0]
    invT = torch.inverse(T)
    # should make sure that original p is the element of degree n...
    vx = torch.concat([x[sym_tn_degs[i]] for i in range(len(sym_tn_degs))]);
    return torch.matmul(invT, vx)
# TODO: Reshape this symmetric matrix format to full matrix... (or just make training data in this form too)

def get_irrep_dims(m, n):
    tnsizes = [(m + n - 2*i - 2) + 1 for i in range(min(m-1,n-1)+1)] # degree is (m + n - 2*i - 2)
    # size one bigger than degree. (degree is m-1, n-1)
    tnorders = [i for i in range(min(m-1,n-1)+1)]
    return tnsizes, tnorders

def compute_induced_from_lie(t, d, basis=0):
    # The three basis elements of sl(2) are:
    # [1, 0; 0, -1], [0 1; 0 0] and [0 0; 1 0]
    d = int(d)
    A = torch.zeros((d+1, d+1))
    if basis == 0:
        for i in range(d+1):
            A[i,i] = torch.exp((d-2*i) * t * torch.tensor(1))
    elif basis == 1:
        for k in range(d+1):
            for j in range(k+1):
                A[k,j] = binom(k,j) * t**(k-j)
    elif basis == 2:
        for k in range(d+1):
            for j in range(k+1):
                A[d-k,d-j] = binom(k,j) * t**(k-j)
    return A

if __name__ == "__main__":
    import doctest
    doctest.testmod()
