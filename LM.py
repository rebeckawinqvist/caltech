from __future__ import print_function

import logging
logger = logging.getLogger(__name__)

from collections import Iterable
import numpy as np
import polytope as pc
from scipy.linalg import block_diag




def createLM(ssys, N, poly_list, P_1=None, P_2=None, disturbance_ind=None):
    """Compute L and M of the polytope:

            L[x(0)' u(0)' ... u(N-1)']' <= M

    for systems described by

            x(t+1) = Ax(t) + Bu(t) + Ed(t)


    @param ssys: system dynamics
    @type ssys: L{LtiSysDyn}

    @param N: horizon length

    @param P_1: starting polytope; x(0), ..., x(N-1) \in P_1
    @type P_1: C{Polytope}
    @param P_2: goal polytope; x(N) \in P_2
    @type P_2: C{Polytope}

    """

    # polytope list
    if not isinstance(poly_list, Iterable):
        poly_list = [poly_list] + (N - 1) * [P_1] + [P_2]

    if disturbance_ind is None:
        disturbance_ind = range(1,N+1)

    # sys dynamics and constraints
    A = ssys.A
    B = ssys.B
    E = ssys.E
    D = ssys.Wset
    U = ssys.Uset
    u_constr = U.b
    u_poly_rows = U.A.shape[0]

    ss_dim = A.shape[1]
    u_dim = B.shape[1]
    dist_dim = E.shape[1]

    # initialize L, M, G
    constr_list = [poly.A.shape[0] for poly in poly_list]
    sum_constr = sum(constr_list)

    L_k = np.zeros([sum_constr, ss_dim + N * u_dim])
    L_u = np.zeros([u_poly_rows * N, ss_dim + N * u_dim])

    M_k = np.zeros([sum_constr, 1])
    M_u = np.tile(u_constr.reshape(u_constr.size, 1), (N, 1))

    G_k = np.zeros([sum_constr, dist_dim * N])
    G_u = np.zeros([u_poly_rows * N, dist_dim * N])

    A_n = np.eye(ss_dim)
    A_k = np.zeros([ss_dim, ss_dim * N])

    B_diag = B
    E_diag = E
    for i in range(N - 1):
        B_diag = block_diag(B_diag, B)
        E_diag = block_diag(E_diag, E)

    poly_row = 0
    # create L, M, G
    for i in range(N + 1):
        poly = poly_list[i]
        constr_dim = poly.A.shape[0]

        ### M ###
        idx = range(poly_row, poly_row + constr_dim)
        M_k[idx, :] = poly.b.reshape(poly.b.size, 1)

        ### G ###
        F = poly.A
        if i in disturbance_ind:  # no disturbances for x(0)
            idx = np.ix_(range(poly_row, poly_row + constr_dim),
                         range(G_k.shape[1]))
            G_k[idx] = F.dot(A_k).dot(E_diag)

        ### L ###
        AB_tri = np.hstack([A_n, A_k.dot(B_diag)])

        idx = np.ix_(range(poly_row, poly_row + constr_dim),
                     range(ss_dim + u_dim * N))

        L_k[idx] = F.dot(AB_tri)

        if i >= N:
            continue

        idx = np.ix_(range(i * u_poly_rows, (i + 1) * u_poly_rows),
                     range(ss_dim + i * u_dim, ss_dim + (i + 1) * u_dim))
        L_u[idx] = U.A

        ### Iterate ###
        poly_row += constr_dim
        A_n = A.dot(A_n)
        A_k = A.dot(A_k)

        idx = np.ix_(range(ss_dim),
                     range(i * ss_dim, (i + 1) * ss_dim))
        A_k[idx] = np.eye(ss_dim)

    # Assemble L and M
    if not np.all(G_k==0):
        G = np.vstack([G_k, G_u])
        D_hat = get_max_extreme(G, D, N)
    else:
        G = np.zeros((G_k.shape[0]+G_k.shape[0], G_k.shape[1]))
        D_hat = np.zeros([sum_constr + u_poly_rows*N, 1])

    L = np.vstack([L_k, L_u])
    M = np.vstack([M_k, M_u])
    M_d = M - D_hat

    return (L, M, G, M_d)



def get_max_extreme(G,D,N):
    """Calculate the array d_hat such that::

        d_hat = max(G*DN_extreme),

    where DN_extreme are the vertices of the set D^N.

    This is used to describe the polytope::

        L*x <= M - G*d_hat.

    Calculating d_hat is equivalen to taking the intersection
    of the polytopes::

        L*x <= M - G*d_i

    for every possible d_i in the set of extreme points to D^N.

    @param G: The matrix to maximize with respect to
    @param D: Polytope describing the disturbance set
    @param N: Horizon length

    @return: d_hat: Array describing the maximum possible
        effect from the disturbance
    """
    D_extreme = pc.extreme(D)
    nv = D_extreme.shape[0]
    dim = D_extreme.shape[1]
    DN_extreme = np.zeros([dim*N, nv**N])

    for i in range(nv**N):
        # Last N digits are indices we want!
        ind = np.base_repr(i, base=nv, padding=N)
        for j in range(N):
            DN_extreme[range(j*dim,(j+1)*dim),i] = D_extreme[int(ind[-j-1]),:]

    d_hat = np.amax(np.dot(G,DN_extreme), axis=1)
    return d_hat.reshape(d_hat.size,1)


























