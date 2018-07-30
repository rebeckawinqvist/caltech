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
    num_of_constr = L.shape[0]
    num_of_dist = G_d.shape[1]

    c_epsilon = np.array([[1.] for i in range(num_of_constr)])
    c_d = np.array([[0.] for i in range(num_of_dist)])
    c = np.vstack((c_epsilon, c_d))
    c = matrix(c)

    eye_eps = np.eye(num_of_constr)
    eye_dist = np.eye(num_of_dist)
    zero_eps = np.zeros((num_of_dist, num_of_constr))
    zero_dist = np.zeros((num_of_constr, num_of_dist))

    # Create A
    eps_Gd = np.hstack((eye_eps, -G_d))
    eps = np.hstack((-eye_eps, zero_dist))
    d_min = np.hstack((zero_eps, -eye_dist))
    d_max = np.hstack((zero_eps, eye_dist))
    A = np.vstack((np.vstack((eps_Gd, eps)), np.vstack((d_min, d_max))))
    A = matrix(A)

    # Create b
    eps_Gd_constr = M-L.dot(x0_u)
    eps_constr = np.zeros((num_of_constr, 1))
    d_min_constr = np.array(D[:,[0]])
    d_max_constr = np.array(D[:,[1]])
    for i in range(N-1):
        d_min_constr = np.vstack((d_min_constr, D[:,[0]]))
        d_max_constr = np.vstack((d_max_constr, D[:, [1]]))
    b = np.vstack((np.vstack((eps_Gd_constr, eps_constr)), np.vstack((d_min_constr, d_max_constr))))
    b = matrix(b)

    # Solve
    sol = solvers.lp(c, A, b)
    print("sol: ", sol['x'])
    d = sol['x'][num_of_constr:, :]
    d = sol['x']

    return d

