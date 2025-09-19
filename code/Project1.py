import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree



# Left side
dx_L = 0.001875
N_L = 320
rho_L = 1.0
P_L   = 1.0
v_L   = 0.0
e_L   = 2.5
# Left arrays
rho_L_arr = np.full(N_L, rho_L)
P_L_arr   = np.full(N_L, P_L)
v_L_arr   = np.full(N_L, v_L)
e_L_arr   = np.full(N_L, e_L)

# Right side
dx_R = 0.0075
N_R = 80
rho_R = 0.25
P_R   = 0.1795
v_R   = 0.0
e_R   = 1.795
# Right arrays
rho_R_arr = np.full(N_R, rho_R)
P_R_arr   = np.full(N_R, P_R)
v_R_arr   = np.full(N_R, v_R)
e_R_arr   = np.full(N_R, e_R)

# Both sides
m = 0.001875
dt = 0.005
gamma = 1.4
n_steps = 40

# Left and Right x
x_L = np.arange(-0.6 + dx_L/2, 0, dx_L)
x_R = np.arange(dx_R/2, 0.6, dx_R)

# Combinatyion of left and right arrays
x = np.concatenate([x_L, x_R])
rho = np.concatenate([rho_L_arr, rho_R_arr])
P   = np.concatenate([P_L_arr, P_R_arr])
v   = np.concatenate([v_L_arr, v_R_arr])
e   = np.concatenate([e_L_arr, e_R_arr])

# Particle mass array (same for all)
m_arr = np.full_like(x, m)

# one array for each x
particles = np.stack([x, rho, P, v, e, m_arr], axis=1)


# Smoothing lengths
eta = 1.2
d = 2
h = np.array(eta * (m_arr/rho)*(1/d))

def avg(arr):
    arr_avg = np.zeros_like(arr)
    arr_avg[1:-1] = 0.5 * (arr[:-2] + arr[2:])
    arr_avg[0] = arr[1] # making sure the first and last item is also averaged
    arr_avg[-1] = arr[-2]
    return arr_avg

h_avg = avg(h)
rho_avg = avg(rho)

c = np.sqrt((gamma - 1) * e)
c_avg = avg(c)

support_radius = 2 * h_avg

p = (gamma -1) *rho_avg*e

plt.plot(x,p)
plt.show()

plt.plot(x,c)
plt.show()


 #####Do I want to use it???######
 # neighbor index for each x
def find_neighbors_kdtree(x):
    tree = cKDTree(x.reshape(-1, 1))
    neighbors = tree.query_ball_point(x.reshape(-1, 1), support_radius)
    return neighbors

neighbors = find_neighbors_kdtree(x)


def cubic_spline_kernel(r, h):
    
    alpha = 1.0 / h
    R = abs(r) / h
    
    # calculating using the R conditions
    if R >= 0 and R < 1:
        W = alpha * (2.0/3.0 - R**2 + 0.5 * R**3)
    elif R >= 1 and R < 2:
        W = alpha * (1.0/6.0) * (2.0 - R)**3
    else:
        W = 0.0
    
    return W


