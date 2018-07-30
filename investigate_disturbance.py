#!/usr/bin/env python
"""Simulation example with continuous dynamics."""
from __future__ import division
from __future__ import print_function

import logging
import random

import numpy as np
import polytope as pc
from polytope import box2poly
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from tulip import hybrid, spec, synth
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition
from tulip.abstract import find_controller
from disturbance_equality_constraints import *
from LM import *
from polytope import *
from find_park_signal import *

logging.basicConfig(level='WARNING')


show = False
# Problem parameters
input_bound = 10.0
uncertainty = 0.01
# Continuous state space
cont_state_space = box2poly([[0., 3.], [0., 2.]])
# Continuous dynamics
A = np.array([[1.0, 0.], [0., 1.0]])
B = np.array([[0.1, 0.], [0., 0.1]])
E = np.array([[1, 0], [0, 1]])
# Available control, possible disturbances
U = input_bound * np.array([[-1., 1.], [-1., 1.]])
W = uncertainty * np.array([[-1., 1.], [-1., 1.]])
D = W
# Convert to polyhedral representation
U = box2poly(U)
W = box2poly(W)
# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)
# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])
# Compute the proposition preserving partition of
# the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)

plot_partition(cont_partition) if show else None
# Given dynamics & proposition-preserving partition,
# find feasible transitions
disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=False,
    conservative=True,
    N=5, min_cell_volume=0.1, plotit=show)

# Visualize transitions in continuous domain (optional)
plot_partition(disc_dynamics.ppp, disc_dynamics.ts,
               disc_dynamics.ppp2ts)
plot_partition(disc_dynamics.ppp, disc_dynamics.ts,
               disc_dynamics.ppp2ts) if show else None
disc_dynamics.plot()
#
# Specification
# Environment variables and assumptions
env_vars = {'park'}
env_init = set()
env_prog = '!park'
env_safe = set()
# System variables and requirements
sys_vars = {'X0reach'}
sys_init = {'X0reach'}
sys_prog = {'home'}  # []<>home
sys_safe = {'(X(X0reach) <-> lot) || (X0reach && !park)'}
sys_prog |= {'X0reach'}
# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
specs.qinit = '\E \A'
#
# Synthesize
ctrl = synth.synthesize(specs,
                        sys=disc_dynamics.ts,
                        ignore_sys_init=True)
assert ctrl is not None, 'unrealizable'
# Generate a graphical representation of the controller for viewing
if not ctrl.save('continuous.png'):
    print(ctrl)
print('\n Simulation starts \n')
T = 10
# let us pick an environment signal
#randParkSignal = [random.randint(0, 1) for b in range(1, T + 1)]
des_trans = [2,0]
#randParkSignal = [random.randint(0, 1) for b in range(1, T + 1)]
randParkSignal = find_sequence(ctrl, des_trans, T)
#randParkSignal = [0 for b in range(1, T+1)]

# initialization:
#     pick initial continuous state consistent with
#     initial controller state (discrete)
u, v, edge_data = ctrl.edges('Sinit', data=True)[1]
s0_part = edge_data['loc']
print('s0_part: ', s0_part)
init_poly_v = pc.extreme(disc_dynamics.ppp[s0_part][0])
x_init = sum(init_poly_v) / init_poly_v.shape[0]
x = [x_init[0]]
y = [x_init[1]]

x_init = 1.5
y_init = 1
print("x: ", x_init)
N = disc_dynamics.disc_params['N']
s0_part = find_controller.find_discrete_state(
    [x_init, y_init], disc_dynamics.ppp)
ctrl = synth.determinize_machine_init(ctrl, {'loc': s0_part})
(s, dum) = ctrl.reaction(1, {'park': randParkSignal[0]})

start = s0_part
end = disc_dynamics.ppp2ts.index(dum['loc'])

u = find_controller.get_input(
    x0=np.array([x_init, y_init]),
    ssys=sys_dyn,
    abstraction=disc_dynamics,
    start=start,
    end=end,
    ord=1,
    mid_weight=5)
# orig: start = s0_part


u_dim = u.shape[1]
u_arr = np.zeros((u_dim*N, 1))
for i in range(u.shape[0]):
    idx = np.ix_(range(i*u_dim, (i+1)*u_dim), range(1))
    u_arr[idx] = u[i,:].reshape(u_dim,1)


x_0 = np.array([[x_init], [y_init]])
x0_u = np.vstack((x_0, u_arr))


regions = disc_dynamics.ppp.regions
p1 = regions[start][0]
p2 = regions[end][0]

L, M, G, M_d = createLM(sys_dyn, N, p1, p1, p2, None)

d = find_greatest_disturbance(L, x0_u, M, G, D, N)

dim = [i+1 for i in range(x_0.shape[0])]
s0_part = projection(Polytope(L, M_d), dim)

import matplotlib as mpl
ax = s0_part.plot(color='red')
ax.set_xlim(0,3)
ax.set_ylim(0,2)
ax.legend()
mpl.pyplot.show()

