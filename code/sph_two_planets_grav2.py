import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Load data
planet_data = np.loadtxt('Planet600.dat')
particles = len(planet_data)

# Define parameters
kappa = 2
gamma = 1.4
h = 5e6
G = 6.67408e-11


# Initial conditions from original data
x = planet_data[:,0]
y = planet_data[:,1]
z = planet_data[:,2]
vx = planet_data[:,3]
vy = planet_data[:,4]
vz = planet_data[:,5]
rho = planet_data[:,7]
p = planet_data[:,8]
e_0 = planet_data[:,8]/((gamma - 1)*planet_data[:,7])       

# Rotation parameters
T1 = 24 * 3600  # Planet 1 24 hours (Earth-like)
T2 = 10 * 3600  # Planet 2 10 hours (faster)
omega1_z = 2 * np.pi / T1  # Angular velocity for planet 1
omega2_z = 2 * np.pi / T2  # Angular velocity for planet 2

# Planet 1 and 2 center and particles
planet1_center_x, planet1_center_y = -5e7, -5e7  
planet2_center_x, planet2_center_y = 5e7, 5e7

# Add rotation to planet 1
vx1_rot = vx + 1000 + (-omega1_z * (y - planet1_center_y))
vy1_rot = vy + 1000 + (omega1_z * (x - planet1_center_x))

# Add rotation to planet 2
vx2_rot = vx - 10000 + (-omega2_z * (y - planet2_center_y))
vy2_rot = vy - 1000 + (omega2_z * (x - planet2_center_x))

# Create two-planet system
S = np.zeros((2*particles, len(planet_data[0])))
S[:,0] = np.reshape(np.array((x - 5e7, x + 5e7)), (particles*2)) # x for the two planets
S[:,1] = np.reshape(np.array((y - 5e7, y + 5e7)), (particles*2)) # y for the two planets
S[:,2] = np.reshape(np.array((z, z)), (particles*2)) # z for the two planets
S[:,3] = np.reshape(np.array((vx1_rot, vx2_rot)), (particles*2)) # vx with rotation
S[:,4] = np.reshape(np.array((vy1_rot, vy2_rot)), (particles*2)) # vy with rotation
S[:,5] = np.reshape(np.array((vz, vz)), (particles*2)) # vz
mass = np.reshape(np.array((planet_data[:,6], planet_data[:,6]*1)), (particles*2))   # mass
S[:,6] = np.reshape(np.array((rho,rho)), (particles*2)) # rho
S[:,7] = np.reshape(np.array((p,p)), (particles*2)) # pressure
S[:,8] = np.reshape(np.array((e_0,e_0)), (particles*2)) # energy

# Calculate initial rotational energy
def calculate_rotational_energy():
    rot_energy = 0
    # Planet 1
    for i in range(particles):
        r_vec = np.array([S[i,0] - planet1_center_x, S[i,1] - planet1_center_y])
        r = np.linalg.norm(r_vec)
        if r > 0:
            I_particle = (4 * np.pi * mass[i] * r**2) / (5 * T1)
            rot_energy += 0.5 * I_particle * omega1_z**2
    
    # Planet 2
    for i in range(particles, 2*particles):
        r_vec = np.array([S[i,0] - planet2_center_x, S[i,1] - planet2_center_y])
        r = np.linalg.norm(r_vec)
        if r > 0:
            I_particle = (4 * np.pi * mass[i] * r**2) / (5 * T2)
            rot_energy += 0.5 * I_particle * omega2_z**2
            
    return rot_energy

initial_rot_energy = calculate_rotational_energy()

'''
plt.style.use('dark_background')
fig = plt.figure(figsize=[12,10])
fig.patch.set_facecolor('black')
ax = fig.add_subplot((111), projection='3d')
ax.set_facecolor('black')


colors1 = np.full(particles, 'cyan')
colors2 = np.full(particles, 'magenta') 
colors = np.concatenate([colors1, colors2])

ax.scatter(S[:,0], S[:,1], S[:,2], c=colors, alpha=0.8, s=15, edgecolors='white', linewidths=0.1)

ax.set_xlim(-2e8, 2e8)
ax.set_xlabel('X [m]', fontsize=12, color='white')
ax.set_ylim(-2e8, 2e8)
ax.set_ylabel('Y [m]', fontsize=12, color='white')
ax.set_zlim(-2e8, 2e8)
ax.set_zlabel('Z [m]', fontsize=12, color='white')
ax.set_title('Initial State: Cyan=Planet1 (24h), Magenta=Planet2 (10h)', 
             fontsize=14, color='white', pad=20)
ax.grid(True, alpha=0.3)
ax.tick_params(colors='white', labelsize=10)
ax.view_init(elev=20, azim=45)  # Set fixed elevation and azimuth angles
ax.mouse_init()  # Reset mouse interactions
fig.canvas.toolbar_visible = False  # Hide toolbar to prevent rotation controls
'''
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

def artificial_viscosity(r, dX, dV, rho_avg, hmean, c_avg, VdotX):
    alpha = 1.5
    beta = 1.5
    
    phi = 0.1*hmean
    
    phi_ij= (hmean*VdotX)/(abs(r)**2 + phi**2)   
    PI = (-alpha*c_avg*phi_ij + beta*phi_ij*phi_ij)/(rho_avg)

    return PI

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
    
    pot_matrix[:,mask_03] = 1/((r[mask_03])**2)*norm[:,mask_03]
    
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
    # Reshape input vector
    S = S.reshape(N, N_total) 
   
    hmean = np.full([N,N], h)
    rii, rij = np.triu_indices(len(hmean), k=1)
    
    # Positions
    dxx_u = (S[:,0].reshape(N, 1) - S[:,0])[rii, rij]
    dxy_u = (S[:,1].reshape(N, 1) - S[:,1])[rii, rij]
    dxz_u = (S[:,2].reshape(N, 1) - S[:,2])[rii, rij]
    
    # Velocities
    dvx_u = (S[:,3].reshape(N, 1) - S[:,3])[rii, rij]
    dvy_u = (S[:,4].reshape(N, 1) - S[:,4])[rii, rij]
    dvz_u = (S[:,5].reshape(N, 1) - S[:,5])[rii, rij]
     
    # Pairs and indecies
    dX_u = np.array((dxx_u, dxy_u, dxz_u))
    dV_u = np.array((dvx_u, dvy_u, dvz_u))
    r_u = np.linalg.norm(dX_u, axis = 0)
    
    # Nearest neighbor mask
    kappa_mask = (r_u <= kappa*hmean[rii, rij]) & (r_u > 0)
    p_i = rii[kappa_mask]
    p_j = rij[kappa_mask]
        
    # Newtonian gravity
    F_i = -G*mass[rij]*(dX_u/(r_u*r_u*r_u))
    F_j =  G*mass[rii]*(dX_u/(r_u*r_u*r_u)) 
    
    
    # Nearest neighbors = 0 
    F_i[:,kappa_mask] = 0
    F_j[:,kappa_mask] = 0
    
    # Interactions of each particle to another
    grav_long = np.zeros((3, N))
    np.add.at(grav_long.T, rii, F_i.T)
    np.add.at(grav_long.T, rij, F_j.T)
       
    # Mask the pairs
    dX = dX_u[:, kappa_mask]
    dV = dV_u[:, kappa_mask]
    hmean =  hmean[rii, rij][kappa_mask]
    r = r_u[kappa_mask]
    
    # Kernel and Gradient
    W = kernel_cubic_spline(r, hmean)
    dW = kernel_gradient(r, dX, hmean)
    
    # Pair quantities 
    rho_avg = ((((S[:,6]).reshape(N, 1) + S[:,6])*0.5)[rii, rij])[kappa_mask]
    c = np.sqrt((gamma - 1)*S[:,8]) # Sound speed    
    c_avg = ((((c.reshape(len(c), 1) + c))*0.5)[rii, rij])[kappa_mask]
    
    # Artificial viscosity
    VdotX = np.sum(dV*dX, axis=0)
    visc_mask = (VdotX < 0) 
    viscosity = np.zeros(len(p_i))
    viscosity[visc_mask] = artificial_viscosity(r[visc_mask], dX[:,visc_mask], dV[:,visc_mask],
                                    rho_avg[visc_mask], hmean[visc_mask], 
                                    c_avg[visc_mask], VdotX[visc_mask])

    return p_i, p_j, W, dW, viscosity, dX, dV, hmean, r, rii, rij, grav_long


def integrate_flat(t, S_flat):
    # Logger
    if int(t*1e-3) % 10 == 0:  
        print(f"Integrating at t = {t:.2f} s")
        
    S = S_flat.reshape(N, N_total)
    
    # Compute sph quantities
    pi_s, pj_s, W, dW, viscosity, dX, dV, hmean, r, rii, rij, grav_long = sph(S_flat)
        
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

    # Adding gravity
    pot_matrix = dphi_dr(r, dX, hmean)
    gravity_i = gravity(mass[pj_s], dX, pot_matrix)
    gravity_j = gravity(mass[pi_s], -dX, pot_matrix)
    
    # Combine SPH and gravity forces for nearest neighbors
    dv_i = gravity_i + velocity_i
    dv_j = gravity_j + velocity_j
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
print("\nStarting simulation with rotating planets...")
S_flat = S.reshape(-1)
t_span = (0, 1*12000)  # total time in seconds
t_eval = np.linspace(t_span[0], t_span[1], 700)
solution = solve_ivp(integrate_flat, t_span, S_flat, method='RK45', t_eval=t_eval, max_step=25.0)

fig2d, ax2d = plt.subplots(figsize=(8,8))

def update_2d(frame):
    ax2d.clear()
    ax2d.set_xlim(-2e8, 2e8)
    ax2d.set_ylim(-2e8, 2e8)
    ax2d.set_xlabel('x [m]')
    ax2d.set_ylabel('y [m]')
    S_frame = solution.y[:, frame].reshape(N, N_total)
    
    # Color by planet
    colors = ['blue'] * particles + ['red'] * particles
    scatter = ax2d.scatter(S_frame[:,0], S_frame[:,1], c=colors, s=5, alpha=0.7)
    ax2d.set_title(f'Rotating Planets: t={solution.t[frame]:.1f} s')
    return scatter,

ani = animation.FuncAnimation(fig2d, update_2d, frames=len(t_eval), interval=50)
ani.save('SPH_2D_rotating_animation.gif', writer='pillow', fps=20)

print("Simulation completed!")


plt.show()