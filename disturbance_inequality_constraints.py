from cvxopt import *
import numpy as np


def find_greatest_disturbance(L, x0_u, M, G_d, D, N):
    """"Find greatest enlargement of D so that the following is still satisfied:

            x(0)...x(N-1) \in pre(PN)/\P0
            x(N) \in PN

        Solves the following optimization problem:

            LP:     min \epsilon
                    s.t. \epsilon = M - L[x(0) u]' - Gd

            LP:     min c^Tx
                    s.t. Ax = b
                    x >= 0

    """
    num_of_dist = G_d.shape[1]

    c = np.array([[-1.] for i in range(num_of_dist)])
    c = matrix(c)

    # Create A
    A = matrix(-G_d)

    # Create b
    constr = M-L.dot(x0_u)
    b = matrix(constr)

    # Solve
    sol = solvers.lp(c, A, b)
    d = sol['x']

    return d

