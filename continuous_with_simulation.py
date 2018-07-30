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
from find_park_signal import *
from disturbance_equality_constraints import *
from LM import *



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
print("W: \n", W[:,[1]])
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

regions = disc_dynamics.ppp.regions

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
randParkSignal = [random.randint(0, 1) for b in range(1, T + 1)]

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
N = disc_dynamics.disc_params['N']
s0_part = find_controller.find_discrete_state(
    [x[0], y[0]], disc_dynamics.ppp)
x0_region = s0_part
ctrl = synth.determinize_machine_init(ctrl, {'loc': s0_part})
(s, dum) = ctrl.reaction('Sinit', {'park': randParkSignal[0]})
print(dum)

u_vec = []
for i in range(0, T):
    (s, dum) = ctrl.reaction(s, {'park': randParkSignal[i]})
    #print('s: ', s)
    #print('park signal: ', randParkSignal[i])
    #print("disc_dynamics: ", disc_dynamics)
    u = find_controller.get_input(
        x0=np.array([x[i * N], y[i * N]]),
        ssys=sys_dyn,
        abstraction=disc_dynamics,
        start=s0_part,
        end=disc_dynamics.ppp2ts.index(dum['loc']),
        ord=1,
        mid_weight=5) #originally: mid_weight = 5
    u_vec.append(u)
    for ind in range(N):
        s_now = np.dot(
            sys_dyn.A, [x[-1], y[-1]]
        ) + np.dot(sys_dyn.B, u[ind])
        x.append(s_now[0])
        y.append(s_now[1])
    s0_part = find_controller.find_discrete_state(
        [x[-1], y[-1]], disc_dynamics.ppp)
    s0_loc = disc_dynamics.ppp2ts[s0_part]
    print('s_loc: ', s0_loc)
    print('dum[loc]: ', dum['loc'], '\n')
    #print(dum)

show_traj = True
if show_traj:
    assert plt, 'failed to import matplotlib'
    plt.plot(x)
    plt.plot(y)
    plt.show()


u_arr = np.array(u_vec)
x_0 = np.array([[x[0], y[0]]])
x0_u = np.vstack((x_0, u_arr))

