from cvxopt import *
import numpy as np

def disturbance_optimization(L, x0_u, M, G_d, D, N):
    """
    Minimize \epsilon / Maximize d so that the following still holds

        \epsilon - Gd = M - L[x(0) u]'
        \epsilon <= 0
        d \in D


    The optimization problem to be solved individually for each element within \epsilon is

            min     \epsilon_i
            s.t.    \epsilon_i - G_id_i = M_i - L_i[x(0) u]'
                    \epsilon_i <= 0
                    d_i \in D


    :param L:
    :param x0_u:
    :param M:
    :param G_d:
    :param D:
    :return: Sequence of disturbances.
    """

    # number of constraints
    n = L.shape[0]
    # number of disturbances
    m = D.shape[1]
    print("m: ", m)
    print("N: ", N)


    # cost function

    # c, G, h
    c = matrix(np.vstack(([1.], np.vstack(([0. for i in range(m*N)])))))

    G = np.zeros((2*m*N+1, m*N+1))
    G[0,0] = -1
    for i in range(2*m*N):
        j = i%(m*N)
        idx = np.ix_(range(i+1, i+2), range(j+1, j+2))
        G[idx] = 1
        if i < m*N:
            G[idx] = -1

    G = matrix(G)

    h_e = 0
    h_lb = np.vstack((-D[:,[0]] for i in range(N)))
    h_ub = np.vstack((D[:,[1]] for i in range(N)))
    h = matrix(np.vstack((np.vstack((h_e, h_lb)), h_ub)))

    eps = []
    d = []

    for i in range(n):
        print("G_d: \n", G_d[i,:])
        A = matrix(np.hstack(([1.], np.array(G_d[i,:])))).T
        print("A: \n", A)
        b = matrix(M[i,:]-L[i,:].dot(x0_u))

        sol = solvers.lp(c, G, h, A, b)
        eps_val = sol['x'][0]
        d_val = list(sol['x'][1:])
        print("d_val: ", d_val)
        eps.append(eps_val)
        d.append(d_val)

    return eps, d
