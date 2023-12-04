# In this file we just construct the polynomial certificates of a fixed degree

import numpy as np
from sympy.functions.special.polynomials import gegenbauer
import cvxpy 

import sympy


def solve_spherical_code(k, alpha, d, solver = "SCS"):
    """
    Finds the polynomial certificate of degree d using k polynomials (up to and including deg k)
    for the code [-1, alpha]
    """

    # First compute f
    g = cvxpy.Variable(k+1)
    constraints = [g >= 0, g[0] == 1]

    # We make the substitution p((x1 * x^2 + x2) / (1 + x^2))
    # to map R to [x1, x2]
    # Here, x1 = -1, x2 = alpha
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    gpolys = [(d + 2 * i - 2) / (d - 2) * sympy.simplify((1+y**2)**(k) * (gegenbauer(i, (d-2)/2, x).replace(x,(-y**2 + alpha)/(1 + y**2)))) for i in range(k+1)]
    gpolys = [np.array(gpolys[i].as_poly().as_list()) for i in range(len(gpolys))]

    # This is f(1) of the original polynomial
    f1 = g[0]+cvxpy.sum([cvxpy.sum(g[i] * ((d + 2 * i - 2) / (d - 2) *np.array(gegenbauer(i, (d-2)/2, x).as_poly().as_list()))) for i in range(1,len(gpolys))])

    # This is f for polynomial which must be < 0 everywhere
    f = cvxpy.sum([g[i] * gpolys[i] for i in range(k+1) ])

    X = cvxpy.Variable((k+1, k+1), symmetric = True)
    for monomial_degree in range(f.shape[0]): # match monomials
        this_term = 0
        for i in range(max(0,monomial_degree - k), min(monomial_degree, k)+1):
            j = monomial_degree - i
            this_term += X[i, j]
        constraints += [this_term == -f[f.shape[0] - 1 - monomial_degree]] # -f >= 0
    constraints += [X >> 0]
    prob = cvxpy.Problem(cvxpy.Minimize(f1), constraints)
    prob.solve(solver = solver, verbose = False)
    return sum([g[i].value * gpolys[i] for i in range(k+1)])

