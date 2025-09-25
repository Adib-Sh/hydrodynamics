import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# Load data
planet_data = np.loadtxt('Planet300.dat')
particles = len(planet_data)

# Define parameters
kappa = 2
gamma = 1.4
h = 5e6
G = 6.67408e-11

# Initial conditions
x = planet_data[:,0]
y = planet_data[:,1]
z = planet_data[:,2]
vx = planet_data[:,3]
vy = planet_data[:,4]
vz = planet_data[:,5]
rho = planet_data[:,7]
p = planet_data[:,8]

'''
    e_0 = planet_data[:,8]/((gamma - 1)*planet_data[:,7])   
    # Initial state vector 
    S = np.zeros((particles, len(planet_data[0])))
    S[:,0] = x
    S[:,1] = y 
    S[:,2] = z
    S[:,3] = vx
    S[:,4] = vy
    S[:,5] = vz
    S[:,6] = rho
    S[:,7] = p
    S[:,8] = e_0
    m = planet_data[:,6]
'''        

e_0 = planet_data[:,8]/((gamma - 1)*planet_data[:,7])       
# Initial state vector 
S = np.zeros((2*particles, len(planet_data[0])))
S[:,0] = np.reshape(np.array((x - 5e7, x + 5e7)), (particles*2)) # x for the two planets
S[:,1] = np.reshape(np.array((y - 5e7, y + 5e7)), (particles*2))# y for the two planets
S[:,2] = np.reshape(np.array((z, z)), (particles*2)) # z for the two planets
S[:,3] = np.reshape(np.array((vx + 1000, vx - 10000)), (particles*2)) # vx
S[:,4] = np.reshape(np.array((vy + 1000, vy - 1000)), (particles*2)) # vy
S[:,5] = np.reshape(np.array((vz, vz)), (particles*2)) # vz
mass = np.reshape(np.array((planet_data[:,6], planet_data[:,6]*100)), (particles*2))   # mass
S[:,6] = np.reshape(np.array((rho,rho)), (particles*2)) # rho
S[:,7] = np.reshape(np.array((p,p)), (particles*2)) # pressure
S[:,8] = np.reshape(np.array((e_0,e_0)), (particles*2)) # energy

fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot((111), projection='3d')
ax.scatter(S[:,0], S[:,1], S[:,2])
ax.set_xlim(-2e8, 2e8)
ax.set_xlabel('$x [m]$')
ax.set_ylim(-2e8, 2e8)
ax.set_ylabel('$y [m]$')
ax.set_zlim(-2e8, 2e8)
ax.set_zlabel('$z [m]$')

# Reshape initial state vector
N = len(S)
N_total = len(S[0])
S = S.reshape(N*N_total) 
hlen = np.full(N, h)


def kernel_cubic_spline(r, hmean):
    alpha_d = (3/(2*np.pi*hmean**3)) # Alpha-d in 3D
    R = r/hmean
    W = np.zeros(len(R))
    
    mask_01 = (R >= 0) & (R < 1)
    mask_02 = (R >= 1) & (R < 2)
  
    W[mask_01] = alpha_d[mask_01]*(2/3 - (R[mask_01])**2 + 0.5*(R[mask_01])**3)
    W[mask_02] = alpha_d[mask_02]*((2-(R[mask_02]))**3)/6   
                  
    return W


def kernel_gradient(r, dX, hmean):
    
    alpha_d = (3/(2*np.pi*hmean**3)) # Alpha-d in 3D
    R = r/hmean
    dW = np.zeros((3, len(R))) 
    
    mask_01 = (R >= 0) & (R < 1)
    mask_02 = (R >= 1) & (R < 2)
      
    dX_1 = dX[:,mask_01]
    dX_2 = dX[:,mask_02]
    
    constant1 = alpha_d[mask_01]*(-2 + 1.5*(R[mask_01]))/(hmean[mask_01]**2)    
    dW[:,mask_01] = constant1*(dX_1)
    
    constant2 = -alpha_d[mask_02]*(0.5*((2-(R[mask_02]))**2))/(hmean[mask_02]*r[mask_02])    
    dW[:,mask_02] = constant2*(dX_2)

    
    return dW


def Pi_ij(r, dX, dV, rho_avg, hmean, c_avg, VdotX):
    alpha = 1
    beta = 1
    phi = 0.1*hmean
    
    phi_ij = (hmean*VdotX)/(abs(r)**2 + phi**2)   
    Pi_ij = (-alpha*c_avg*phi_ij + beta*(phi_ij*phi_ij))/(rho_avg)

    return Pi_ij


def dphi_dr(r, dX, hmean):
    R = r/hmean
    
    mask_01 = (R >= 0) & (R < 1)
    mask_02 = (R >= 1) & (R < 2)
    mask_03 = (R >= 2)
    
    pot_matrix = np.zeros((3, (len(dX[0]))))
    norm = dX/r
    
    pot_matrix[:,mask_01] = (1/((hmean[mask_01] + 0.1)**2))*((4*R[mask_01])/3 
             - (6*(R[mask_01]**3)/5) + 0.5*R[mask_01]**4)*norm[:,mask_01]
    
    pot_matrix[:,mask_02] = (1/((hmean[mask_02])**2))*((8/3)*R[mask_02] - 3*R[mask_02]**2 
             + (6/5)*R[mask_02]**3 - (1/6)*R[mask_02]**4 - 1/(15*(R[mask_02])**2))*norm[:,mask_02]
    
    pot_matrix[:,mask_03] = 1/((r[mask_03])**2)**norm[:,mask_03]
    
    return pot_matrix


def gravity(m, dX, pot_matrix):
   
    r = np.linalg.norm(dX, axis=0)
    g = -0.5*G*m*(2*pot_matrix)*(dX/r)
    
    return g
    

def velocity(kernel_gradient, m, p1, p2, rho_1, rho_2, Pi_ij):
    
    ratio = (p1/(rho_1*rho_1)) + (p2/(rho_2*rho_2))
    d_V = -m*(ratio + Pi_ij)*kernel_gradient
    
    return d_V


def energy(mass, rho_1, rho_2, p1, p2, dV, kernel_gradient, Pi_ij):
    
    return 0.5*mass*((p1/(rho_1*rho_1)) + (p2/(rho_2*rho_2)) 
                     + Pi_ij)*np.sum(dV*kernel_gradient, axis=0) 
    

def sph(S):
           
#    S[:,0] = x
#    S[:,1] = y
#    S[:,2] = z
#    S[:,3] = vx
#    S[:,4] = vy
#    S[:,5] = vz
#    S[:,6] = density rho
#    S[:,7] = pressure
#    S[:,8] = energy
    
    # Reshape input vector
    S = S.reshape(N, N_total) 
   
    hmean = np.full([N,N], h)

    trii, trij = np.triu_indices(len(hmean), k=1)
    
    # Positions
    dxx_u = (S[:,0].reshape(N, 1) - S[:,0])[trii, trij]
    dxy_u = (S[:,1].reshape(N, 1) - S[:,1])[trii, trij]
    dxz_u = (S[:,2].reshape(N, 1) - S[:,2])[trii, trij]
    
    # Velocities
    dvx_u = (S[:,3].reshape(N, 1) - S[:,3])[trii, trij]
    dvy_u = (S[:,4].reshape(N, 1) - S[:,4])[trii, trij]
    dvz_u = (S[:,5].reshape(N, 1) - S[:,5])[trii, trij]
     
    # Pairs and indecies
    dX_u = np.array((dxx_u, dxy_u, dxz_u))
    dV_u = np.array((dvx_u, dvy_u, dvz_u))
    r_u = np.linalg.norm(dX_u, axis = 0)
    
    # Nearest neighbor mask
    kappa_mask = (r_u <= kappa*hmean[trii, trij]) & (r_u > 0)

    p_i = trii[kappa_mask]
    p_j = trij[kappa_mask]
    

        
    # Cheking the lower side of the matrix and neighbors
    longrange_i = -G*mass[trij]*(dX_u/(r_u*r_u*r_u))
    longrange_j =  G*mass[trii]*(dX_u/(r_u*r_u*r_u)) 
    
    # <Nearest neighbors = 0 
    longrange_i[:,kappa_mask] = 0
    longrange_j[:,kappa_mask] = 0
    
    # Interactions of each particle to another
    grav_long = np.zeros((3, N))
    np.add.at(grav_long.T, trii, longrange_i.T)
    np.add.at(grav_long.T, trij, longrange_j.T)
    
       
    # Mask the pairs
    dX = dX_u[:, kappa_mask]
    dxx, dxy, dxz = dX
    
    dV = dV_u[:, kappa_mask]
    dvx, dvy, dvz = dV
    
    hmean =  hmean[trii, trij][kappa_mask]
    r = r_u[kappa_mask]
    
    # Kernel and Gradient
    W = kernel_cubic_spline(r, hmean)
    dW = kernel_gradient(r, dX, hmean)
    
    # Pair quantities 
    rho_avg = ((((S[:,6]).reshape(N, 1) + S[:,6])*0.5)[trii, trij])[kappa_mask]
    c = np.sqrt((gamma - 1)*S[:,8]) # Sound speed    
    c_avg = ((((c.reshape(len(c), 1) + c))*0.5)[trii, trij])[kappa_mask]
    
    # Artificial viscosity
    VdotX = np.sum(dV*dX, axis=0)
    maskVISC = (VdotX < 0) 
    viscosity = np.zeros(len(p_i))
    viscosity[maskVISC] = Pi_ij(r[maskVISC], dX[:,maskVISC], dV[:,maskVISC],
             rho_avg[maskVISC], hmean[maskVISC], c_avg[maskVISC], VdotX[maskVISC])

    
    return p_i, p_j, W, dW, viscosity, dX, dV, hmean, r, trii, trij, grav_long


def integrate_flat(t, S_flat):
    # Logger
    if int(t*1e-3) % 10 == 0:  
        print(f"Integrating at t = {t:.2f} s")
        
    S = S_flat.reshape(N, N_total)
    
    # Compute sph quantities
    pi_s, pj_s, W, dW, viscosity, dX, dV, hmean, r, trii, trij, grav_long = sph(S_flat)
        
    dS = np.zeros_like(S)

    # Density
    S[:,6] = mass / (np.pi * hlen**3)  # self-contribution
    density_i = mass[pj_s] * W
    density_j = mass[pi_s] * W
    np.add.at(S[:,6], pi_s, density_i)
    np.add.at(S[:,6], pj_s, density_j)

    # Pressure
    S[:,7] = (gamma - 1) * S[:,6] * S[:,8]

    # Velocity
    velocity_i = velocity(dW, mass[pj_s], S[:,7][pi_s], S[:,7][pj_s], S[:,6][pi_s], S[:,6][pj_s], viscosity)
    velocity_j = -velocity(dW, mass[pi_s], S[:,7][pj_s], S[:,7][pi_s], S[:,6][pj_s], S[:,6][pi_s], viscosity)


    pot_matrix = dphi_dr(r, dX, hmean)
    dv_i = -G*mass[pj_s]*pot_matrix + velocity_i
    dv_j =  G*mass[pi_s]*pot_matrix + velocity_j
    np.add.at(grav_long.T, pi_s, dv_i.T)
    np.add.at(grav_long.T, pj_s, dv_j.T)


    # Derivatives
    dS[:,0] = S[:,3]  # dx/dt = vx
    dS[:,1] = S[:,4]  # dy/dt = vy
    dS[:,2] = S[:,5]  # dz/dt = vz
    dS[:,3] = grav_long[0]  # dvx/dt
    dS[:,4] = grav_long[1]  # dvy/dt
    dS[:,5] = grav_long[2]  # dvz/dt

    # Energy derivatives
    energy_i = energy(mass[pj_s], S[:,6][pi_s], S[:,6][pj_s], S[:,7][pi_s], S[:,7][pj_s], dV, dW, viscosity)
    energy_j = energy(mass[pi_s], S[:,6][pj_s], S[:,6][pi_s], S[:,7][pj_s], S[:,7][pi_s], dV, dW, viscosity)
    np.add.at(dS[:,8], pi_s, energy_i)
    np.add.at(dS[:,8], pj_s, energy_j)

    # Density and pressure derivatives
    dS[:,6] = 0
    dS[:,7] = 0

    return dS.reshape(-1)  # flatten for solve_ivp

# Solver
S_flat = S.reshape(-1)
t_span = (0, 1*2000)  # total time in seconds
t_eval = np.linspace(t_span[0], t_span[1], 200)
solution = solve_ivp(integrate_flat, t_span, S_flat, method='RK45', t_eval=t_eval, max_step=25.0)


# Plots
fig2d, ax2d = plt.subplots(figsize=(8,8))
ax2d.set_xlim(-2e8, 2e8)
ax2d.set_ylim(-2e8, 2e8)
ax2d.set_xlabel('x [m]')
ax2d.set_ylabel('y [m]')
scat = ax2d.scatter([], [], c=[], s=5, cmap='hot')

def update_2d(frame):
    ax2d.clear()
    ax2d.set_xlim(-2e8, 2e8)
    ax2d.set_ylim(-2e8, 2e8)
    ax2d.set_xlabel('x [m]')
    ax2d.set_ylabel('y [m]')
    S_frame = solution.y[:, frame].reshape(N, N_total)
    scatter = ax2d.scatter(S_frame[:,0], S_frame[:,1], c=S_frame[:,6], s=5, cmap='hot')
    ax2d.set_title(f'Time: {solution.t[frame]:.1f} s')
    return scatter,

ani = animation.FuncAnimation(fig2d, update_2d, frames=len(t_eval), interval=50)

# Save the animation as GIF
ani.save('SPH_2D_animation.gif', writer='pillow', fps=20)

# Save a snapshot figure at the final time
S_final = solution.y[:,-1].reshape(N, N_total)
plt.figure(figsize=(8,8))
plt.scatter(S_final[:,0], S_final[:,1], c=S_final[:,6], s=5, cmap='hot')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(f'2D SPH final state at t={solution.t[-1]:.1f} s')
plt.colorbar(label='Density')
plt.savefig('SPH_2D_final.png', dpi=300)
plt.show()


