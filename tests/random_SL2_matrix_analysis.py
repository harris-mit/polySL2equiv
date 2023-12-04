import numpy as np
import math
import torch
from helper import get_SL2_matrices, induced
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def get_cond_numbers(matarray):
    print('matarray', type(matarray))
    print(matarray.shape)
    if type(matarray) == type(np.random.rand(3)):
        conds = []
        for i in range(matarray.shape[0]):
            conds.append(np.linalg.cond(matarray[i]))
        return np.array(conds)
    else:
        conds = []
        for i in range(matarray.shape[0]):
            conds.append(torch.linalg.cond(matarray[i]))
        return np.array(conds)

def get_norms(matarray):
    print('matarray', type(matarray))
    print(matarray.shape)
    if type(matarray) == type(np.random.rand(3)):
        conds = []
        for i in range(matarray.shape[0]):
            conds.append(np.linalg.norm(matarray[i], ord=2))
        return np.array(conds)
    else:
        conds = []
        for i in range(matarray.shape[0]):
            conds.append(torch.linalg.norm(matarray[i], ord=2))
        return np.array(conds)

fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)

# Look at conditioning from gaussian vs iwasawa

nummatrices = 100
print('Getting Gaussian-generated matrices')
gaussian_As = get_SL2_matrices(nummatrices, return_tensor=False, mode='gaussian', thresh=1e50) # no filtering
print('Getting Iwasawa-generated matrices')
iwasawa_As = get_SL2_matrices(nummatrices, return_tensor=False, mode='iwasawa', thresh=1e50) # no filtering
gaussian_conds, iwasawa_conds = get_cond_numbers(gaussian_As), get_cond_numbers(iwasawa_As)

plt.figure()
plt.plot(np.log10(gaussian_conds), 'o')
plt.title('Gaussian log10 conds')
plt.savefig(os.path.join(fig_dir, 'gaussian_conds.pdf'))

plt.figure()
plt.plot(np.log10(iwasawa_conds), 'o')
plt.title('Iwasawa log10 conds')
plt.savefig(os.path.join(fig_dir, 'iwasawa_conds.pdf'))

# -------------- relation between cond(A), d, and cond(A^[d])

dmax = 6
dvals = range(dmax)
Ainduced_conds = np.zeros((len(dvals), nummatrices))
for d in dvals:
    for Aind in range(nummatrices):
        A = gaussian_As[Aind,...]
        Ainduced_conds[d, Aind] = np.linalg.cond(induced(A, k=d, scaled=False, return_tensor=False))

    # plot cond(A) vs cond(A^[d]) for each d
    plt.figure()
    plt.plot(np.log10(gaussian_conds), np.log10(Ainduced_conds[d, :]), 'o')
    plt.xlabel('log10 Cond(A)')
    plt.ylabel('log10 Cond(A^[d])')
    plt.savefig(os.path.join(fig_dir, f'cond_A_vs_induced_d_{d}.pdf'))

for Aind in range(4):
    # plot d vs cond(A^[d]) for a few A
    plt.figure()
    plt.plot(dvals, np.log10(Ainduced_conds[:, Aind]), 'o')
    plt.xlabel('d')
    plt.ylabel('log10 Cond(A^[d])')
    plt.savefig(os.path.join(fig_dir, f'd_vs_induced_cond_example_{Aind}.pdf'))

# same, but with norms

gaussian_norms = get_norms(gaussian_As)
dmax = 6
dvals = range(dmax)
Ainduced_norms = np.zeros((len(dvals), nummatrices))
for d in dvals:
    for Aind in range(nummatrices):
        A = gaussian_As[Aind,...]
        Ainduced_norms[d, Aind] = np.linalg.norm(induced(A, k=d, scaled=False, return_tensor=False), ord=2)
        condnum = np.linalg.cond(induced(A, k=d, scaled=False, return_tensor=False))
        print('norm', Ainduced_norms[d, Aind], 'condition number', condnum, 'divison', Ainduced_norms[d, Aind] / condnum)

    # plot cond(A) vs cond(A^[d]) for each d
    plt.figure()
    plt.plot(np.log10(gaussian_norms), np.log10(Ainduced_norms[d, :]), 'o')
    plt.xlabel('log10 norm(A)')
    plt.ylabel('log10 norm(A^[d])')
    plt.savefig(os.path.join(fig_dir, f'norm_A_vs_induced_d_{d}.pdf'))

for Aind in range(4):
    # plot d vs cond(A^[d]) for a few A
    plt.figure()
    plt.plot(dvals, np.log10(Ainduced_norms[:, Aind]), 'o')
    plt.xlabel('d')
    plt.ylabel('log10 norm(A^[d])')
    plt.savefig(os.path.join(fig_dir, f'd_vs_induced_norm_example_{Aind}.pdf'))






