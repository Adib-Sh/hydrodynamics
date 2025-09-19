import numpy as np
from matplotlib import pyplot as plt

# import SPHSolver

## initial conditions

gamma = 1.4
alpha = 1.0
beta = 1.0
eta = 1.2

dt = 0.005
n_steps = 100

m = 0.001875 # kg
x_min = -0.6 # m
x_max = 0.6 # m

N_hi = 320+1
N_lo = 80

p_hi = 1 # Pa
p_lo = 0.1795 # Pa

rho_hi = 1 # kg/m
rho_lo = 0.25 # kg/m

v_hi = 0.0
v_lo = 0.0

e_hi = 2.5
e_lo = 1.795


N = N_hi+N_lo

# we set evenly spaced particle coordinates
# (the spacing is found in the manual)
# note the +1 for num of steps (becauase Python)

x_hi = np.linspace(-0.6, 0, num=N_hi)
x_lo = np.linspace(0.007500, N_lo*0.007500, num=N_lo)

# initial coordinates (note lower case x_0, this is not state vector X_0)
x_0 = np.concatenate([x_hi, x_lo])

# initial pressure and density
p_0 = np.concatenate([np.full(N_hi, p_hi), np.full(N_lo, p_lo)])
rho_0 = np.concatenate([np.full(N_hi, rho_hi), np.full(N_lo, rho_lo)])
v_0 = np.concatenate([np.full(N_hi, v_hi), np.full(N_lo, v_lo)])
e_0 = np.concatenate([np.full(N_hi, e_hi), np.full(N_lo, e_lo)])
## the state vector shall contain:
## x_i = [x_0, 0, 0] , v_i = 0, m_i, rho_i, p_i

## contsruct the initial state vector X_0

X_0 = np.zeros((N, 11))

X_0[:, 0] = x_0 # set pos
X_0[:, 6] = np.full(N, m)
X_0[:, 7] = rho_0
X_0[:, 8] = p_0
X_0[:, 9] = v_0
X_0[:, 10] = e_0

plt.plot(X_0[:, 0], X_0[:, -1])

# solver = SPHSolver()