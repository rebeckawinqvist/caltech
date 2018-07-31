from cvxopt import *
import numpy as np


def find_greatest_disturbance(L, x0_u, M, G_d, D, N, eps_max=-0.1):
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

    c_z = np.array([1.])
    c_epsilon = np.hstack([0.] for i in range(num_of_constr))
    c_d = np.hstack([0.] for i in range(num_of_dist))
    c = np.hstack((c_z, np.hstack((c_epsilon, c_d))))
    c = matrix(c)

    eye_eps = np.eye(num_of_constr)
    eye_dist = np.eye(num_of_dist)
    zero_eps = np.zeros((num_of_dist, num_of_constr))
    zero_dist = np.zeros((num_of_constr, num_of_dist))
    zero_z_e = np.zeros((eye_eps.shape[0],1))
    zero_z_d = np.zeros((eye_dist.shape[0],1))

    # Create A
    eps_Gd_A = np.hstack((zero_z_e, np.hstack((eye_eps, -G_d))))
    eps_A = np.hstack((zero_z_e, np.hstack((-eye_eps, zero_dist))))
    d_min_A = np.hstack((zero_z_d, np.hstack((zero_eps, -eye_dist))))
    d_max_A = np.hstack((zero_z_d, np.hstack((zero_eps, eye_dist))))
    z_A_row = np.hstack(([-1.], np.array([0. for i in range(num_of_constr+num_of_dist)])))
    z_A = z_A_row
    for i in range(num_of_constr-1):
        z_A = np.vstack((z_A, z_A_row))
    A = np.vstack((np.vstack((np.vstack((eps_Gd_A, z_A)), eps_A)), np.vstack((d_min_A, d_max_A))))
    A = np.vstack((z_A, A))
    A = matrix(A)

    # Create b
    eps_Gd_b = M-L.dot(x0_u)
    eps_b = np.vstack([eps_max] for i in range(num_of_constr))
    d_min_b = np.array(D[:,[0]])
    d_max_b = np.array(D[:,[1]])
    z_b = np.array([[eps_max]])
    for i in range(z_A.shape[0]-1):
        z_b = np.vstack((z_b, np.array([[eps_max]])))
    for i in range(N-1):
        d_min_b = np.vstack((d_min_b, D[:,[0]]))
        d_max_b = np.vstack((d_max_b, D[:, [1]]))
    b = np.vstack((np.vstack((np.vstack((eps_Gd_b, z_b)), eps_b)), np.vstack((d_min_b, d_max_b))))
    print("b: ", b.shape)
    print("z: ", z_b.shape)
    b = np.vstack((z_b, b))
    print(b.shape)
    b = matrix(b)


    # Solve
    sol = solvers.lp(c, A, b)
    print("sol: ", sol['x'])
    d = sol['x']

    return d

