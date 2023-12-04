# File basis_change.py
# Includes the code required to implement a basis change
import sympy
from sympy.abc import x, y, t
import numpy as np
import scipy.integrate as integrate
from generate_data import induced
import torch
from utils import multiply_poly_tensors

def get_fourier2poly(device=torch.device('cpu')):
    """
    Get the linear map that takes
    a, b representing a * cos(kx) + b * sin(kx)
    to the c_i in sum_i c_i * cos(x)^i * sin(x)^(k-i)

    Sympy was painful in returning the coefficients, so I 
    evaluated some in Mathematica and pasted them here. 
    This means there is a limit to the degree that we can handle.
    The first column is TrigExpand[Cos[k*x]] in the basis
    [sin(x)^k, ..., cos(x)^k]. The second column is the same with
    TrigExpand[Sin[k*x]].

    !! Only implemented these identities until degree 8!
    """
    whichT = {}
    whichT[0] = torch.tensor([[1., 0.]]).to(device)
    whichT[1] = torch.tensor([[0, 1.],[1, 0]]).T.to(device)
    whichT[2] = torch.tensor([[-1., 0, 1],[0, 2, 0]]).T.to(device)
    whichT[3] = torch.tensor([[0, -3, 0, 1],[-1., 0, 3, 0]]).T.to(device)
    whichT[4] = torch.tensor([[1., 0, -6, 0, 1],[0, -4, 0, 4, 0]]).T.to(device)
    whichT[5] = torch.tensor([[0., 5, 0, -10, 0, 1],[1, 0, -10, 0, 5, 0]]).T.to(device)
    whichT[6] = torch.tensor([[-1., 0, 15, 0, -15, 0, 1],[0, 6, 0, -20, 0, 6, 0]]).T.to(device)
    whichT[7] = torch.tensor([[0., -7, 0, 35, 0, -21, 0, 1],[-1., 0, 21, 0, -35, 0, 7, 0]]).T.to(device)
    whichT[8] = torch.tensor([[1., 0, -28, 0, 70, 0, -28, 0, 1],[0, -8, 0, 56, 0, -56, 0, 8, 0]]).T.to(device)
    return whichT

def get_fourier2poly_full(device=torch.device('cpu')):
    """
    Get the linear map that takes
    (a_j+ib_j) representing sum_j^k a_j * cos(jx) + b_j * sin(jx)
    to the c_i in sum_i c_i * cos(x)^i * sin(x)^(k-i)

    Sympy was painful in returning the coefficients, so I 
    evaluated some in Mathematica and pasted them here. 
    This means there is a limit to the degree that we can handle.
    Each entry in the dictionary is a tuple of two matrices.
    The first matrix is TrigExpand[Cos[0*x]] in the basis
    [sin(x)^k, ..., cos(x)^k]. The second column is the same with
    TrigExpand[Cos[1*x]], TrigExpand[Cos[2*x]],etc. The second matrix is
    the same thing for TrigExpand[Sin[j*x]]...

    !! Only implemented these identities until degree 8!
    """
    whichT = {k : (get_fourierinhomog(k, 0, device=device), get_fourierinhomog(k, 1, device=device)) for k in [0,2,4,6,8]}
    return whichT

def get_fourierinhomog(k, im = 0, device=torch.device('cpu')):
    """
    If im = 0, get Cos[jx] for j = 0:2:k in homog basis
    If im = 1, get Sin[jx] in homog basis
    (One j per column)
    """
    result = []
    f2p = get_fourier2poly()
    for j in range(0,k,2):
        # image of degree j
        tmp = f2p[j][:,im]
        # boost to degree k
        fancyone = torch.tensor([1.,0,1]) #cos^2 + sin^2
        for i in range(0, (k-j)//2 - 1):
            fancyone = multiply_poly_tensors(fancyone, torch.tensor([1.,0,1]))
        result += [multiply_poly_tensors(tmp, fancyone)]
    result += [f2p[k][:,im]]
    return torch.stack(result).T.to(device)



def get_poly2fourier(d):
    """
    Get the T = poly2fourier basis matrix
    d is the degree of the input polynomial

    T * polynomial = fourier

    >>> d = 6
    >>> p = torch.rand(d+1)
    >>> F = torch.tensor(get_poly2fourier(d))
    >>> th = np.pi/8
    >>> g = np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])
    >>> gd = induced(g, d, return_tensor = True)
    >>> gp = gd.T @ p
    >>> Fgp = F @ torch.complex(gp.double(), torch.zeros(gp.shape).double())
    >>> Fp = F @ torch.complex(p.double(), torch.zeros(p.shape).double())
    >>> rotFp = Fp * np.array([np.exp(j * 1j * th) for j in range(d+1)])
    >>> torch.linalg.norm(rotFp - Fgp) < 10**-7
    tensor(True)
    """
    T = np.zeros((d+1, d+1), dtype = np.cfloat) #cdouble)
    for i in range(0, d+1):
        for j in range(0, d+1):
            cosint = integrate.quad(lambda x: np.cos(x)**i * np.sin(x)**(d-i) * np.cos(j * x), 0, 2*np.pi)[0]/(np.pi)
            sinint = integrate.quad(lambda x: np.cos(x)**i * np.sin(x)**(d-i) * np.sin(j * x), 0, 2*np.pi)[0]/(np.pi)
            #T[j,i] = cosint - 1j * sinint # This would be the true Fourier transform
            T[j,i] = cosint + 1j * sinint # This maintains consistency with the "direction"
            # that we need for equivariance with the Canonical Gram matrix.
            if j == 0:
                T[j, i] /= 2
    return T
