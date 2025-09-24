import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation


# Parameters
gamma = 1.4
alpha = 1
beta = 1

m = 0.001875

# Initial conditions
rho_L, v_L, e_L, p_L = 1.0, 0.0, 2.5, 1.0
dx_L = 0.001875

rho_R, v_R, e_R, p_R = 0.25, 0.0, 1.795, 0.1795
dx_R = 0.0075


def load_planet_data(filename):
    data = np.loadtxt(filename)
    
    particles = {
        'x': data[:, 0:3],      # positions
        'v': data[:, 3:6],      # velocities
        'm': data[:, 6],        # masses
        'rho': data[:, 7],      # densities
        'e': data[:, 8]          # energy
    }
    
    return particles

planet_data = load_planet_data('planet300.dat')
N_total = len(planet_data['x'])



def kernel_cubic_spline(r, h):
    """Cubic spline kernel for scalar distance r"""
    R = r / h
    alpha_d = 3.0 / (2.0 * np.pi * h**3)  # 3D normalization
    
    W = np.zeros_like(R)
    
    # Condition 0 ≤ R < 1
    mask1 = (R >= 0) & (R < 1)
    W[mask1] = alpha_d * (2/3 - R[mask1]**2 + 0.5 * R[mask1]**3)
    
    # Condition 1 ≤ R < 2  
    mask2 = (R >= 1) & (R < 2)
    W[mask2] = alpha_d * (1/6) * (2 - R[mask2])**3
    
    return W

def kernel_gradient(r, h):
    """Kernel gradient for scalar distance r"""
    R = r / h
    alpha_d = 3.0 / (2.0 * np.pi * h**3)
    
    dW_dr = np.zeros_like(R)
    
    # Condition 0 ≤ R < 1
    mask1 = (R >= 0) & (R < 1)
    dW_dr[mask1] = alpha_d * (-2*R[mask1] + 1.5*R[mask1]**2) / h
    
    # Condition 1 ≤ R < 2
    mask2 = (R >= 1) & (R < 2)
    dW_dr[mask2] = -alpha_d * 0.5 * (2 - R[mask2])**2 / h
    
    # Handle division by zero for r=0
    dW_dr = np.where(r > 0, dW_dr / r, 0.0)
    
    return dW_dr

def p(gamma, rho, e):
    return (gamma - 1.0) * rho * e
    
def sound_speed(e):
    return np.sqrt((gamma - 1.0) * np.maximum(e, 1e-10))

def artificial_viscosity(v_i, v_j, dx_ij, rho_i, rho_j, c_i, c_j, h):
    """Calculate artificial viscosity between particle i and j"""
    v_ij = v_j - v_i  # velocity difference
    r_ij = np.linalg.norm(dx_ij)  # distance
    
    # Dot product for compression detection
    v_dot_x = np.dot(v_ij, dx_ij)
    
    # Only apply viscosity for approaching particles
    if v_dot_x >= 0:
        return 0.0
    
    # Calculate viscosity term
    phi_ij = (h * v_dot_x) / (r_ij**2 + (0.1 * h)**2)
    
    rho_avg = 0.5 * (rho_i + rho_j)
    c_avg = 0.5 * (c_i + c_j)
    
    Pi_ij = (-alpha * c_avg * phi_ij + beta * phi_ij**2) / rho_avg
    
    return Pi_ij

# Number of time steps
dt = 0.005
n_steps = 40
t_span = (0, dt * n_steps)
t_eval = np.linspace(t_span[0], t_span[1], n_steps)

def sph(t, y):
    N = N_total

    # --- Reshape y to match original particle data ---
    x = y[0:3*N].reshape(N, 3)        # positions (N, 3)
    v = y[3*N:6*N].reshape(N, 3)      # velocities (N, 3)
    rho = y[6*N:7*N]                  # densities (N,)
    e = y[7*N:8*N]                    # energy (N,)

    h = 1e7  # Smoothing length - you may need to adjust this

    # Create pairwise distance matrix using broadcasting
    dx_matrix = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    r_matrix = np.linalg.norm(dx_matrix, axis=2)  # (N, N)
    
    # Apply cutoff at 2h to avoid unnecessary calculations
    mask_neighbors = r_matrix < 2*h
    r_matrix[~mask_neighbors] = 2*h  # Set large distances to 2h to avoid kernel issues
    
    # Calculate kernel and gradient for all pairs
    W_matrix = kernel_cubic_spline(r_matrix, h)  # (N, N)
    grad_W_scalar = kernel_gradient(r_matrix, h)  # (N, N)
    
    # Convert scalar gradient to vector gradient
    mask_nonzero = r_matrix > 1e-10
    grad_W_vector = np.zeros_like(dx_matrix)
    grad_W_vector[mask_nonzero] = (grad_W_scalar[mask_nonzero, np.newaxis] * 
                                  dx_matrix[mask_nonzero] / r_matrix[mask_nonzero, np.newaxis])
    
    # Calculate density using broadcasting with constant masses
    rho_new = np.sum(planet_data['m'][np.newaxis, :] * W_matrix, axis=1)  # (N,)
    
    # Scale energies to avoid numerical issues - they're too large!
    e_scaled = e * 1e-13  # Scale down by 10^13
    
    # Calculate pressure and sound speed with scaled energies
    p_values = p(gamma, rho_new, e_scaled)  # (N,)
    c_values = sound_speed(e_scaled)  # (N,)
    
    # Prepare arrays for broadcasting
    rho_i = rho_new[:, np.newaxis]  # (N, 1)
    rho_j = rho_new[np.newaxis, :]  # (1, N)
    p_i = p_values[:, np.newaxis]   # (N, 1)
    p_j = p_values[np.newaxis, :]   # (1, N)
    c_i = c_values[:, np.newaxis]   # (N, 1)
    c_j = c_values[np.newaxis, :]   # (1, N)
    
    # Calculate velocity differences
    v_matrix = v[:, np.newaxis, :] - v[np.newaxis, :, :]  # (N, N, 3)
    
    # Artificial viscosity calculation
    v_dot_x = np.sum(v_matrix * dx_matrix, axis=2)  # (N, N)
    mask_compression = (v_dot_x < 0) & mask_neighbors  # Only for approaching neighbors
    
    phi_ij = np.zeros_like(v_dot_x)
    phi_ij[mask_compression] = (h * v_dot_x[mask_compression] / 
                               (r_matrix[mask_compression]**2 + (0.1 * h)**2))
    
    rho_avg = 0.5 * (rho_i + rho_j)  # (N, N)
    c_avg = 0.5 * (c_i + c_j)        # (N, N)
    
    Pi_ij = np.zeros_like(v_dot_x)
    Pi_ij[mask_compression] = (-alpha * c_avg[mask_compression] * phi_ij[mask_compression] + 
                              beta * phi_ij[mask_compression]**2) / rho_avg[mask_compression]
    
    # Calculate pressure term
    pressure_term = p_i / rho_i**2 + p_j / rho_j**2 + Pi_ij[:, :, np.newaxis]  # (N, N, 1)
    
    # Mass matrix for broadcasting - use constant masses
    m_matrix = planet_data['m'][np.newaxis, :, np.newaxis]  # (1, N, 1)
    
    # Acceleration calculation - only sum over neighbors
    dv_dt = -np.sum(m_matrix * pressure_term * grad_W_vector * mask_neighbors[:, :, np.newaxis], axis=1)  # (N, 3)
    
    # Energy calculation
    v_diff_dot_grad = np.sum(v_matrix * grad_W_vector, axis=2)  # (N, N)
    de_dt = 0.5 * np.sum(m_matrix[:, :, 0] * pressure_term[:, :, 0] * v_diff_dot_grad * mask_neighbors, axis=1)  # (N,)
    
    # Scale back the energy derivative
    de_dt_scaled = de_dt * 1e-13  # Scale back up
    
    # Position derivative
    dx_dt = v.copy()  # (N, 3)
    
    # Density derivative - zero since density is calculated by summation
    drho_dt = np.zeros_like(rho_new)  # (N,)
    
    # Pack derivatives
    dy_dt = np.concatenate([
        dx_dt.ravel(),      # 3N
        dv_dt.ravel(),      # 3N  
        drho_dt.ravel(),    # N
        de_dt_scaled.ravel()  # N (scaled back)
    ])
    
    return dy_dt

# Initial state vector
y0 = np.concatenate([
    planet_data['x'].ravel(),    # 3N
    planet_data['v'].ravel(),    # 3N  
    planet_data['rho'].ravel(),  # N
    planet_data['e'].ravel()     # N
])

print(f"Initial state vector size: {len(y0)}")
print(f"Expected size: 8 * N_total = {8 * N_total}")

# Solve the ODE with smaller time steps for stability
sol = solve_ivp(sph, t_span, y0, t_eval=t_eval, method='RK45', 
                max_step=dt/100, rtol=1e-8, atol=1e-10)

# Extract solution for the last time step
final_state = sol.y[:, -1]
N = N_total
x_sol = final_state[0:3*N].reshape(N, 3)
v_sol = final_state[3*N:6*N].reshape(N, 3)
rho_sol = final_state[6*N:7*N]
e_sol = final_state[7*N:8*N]
p_sol = p(gamma, rho_sol, e_sol)

print(f"Solution shape - x: {x_sol.shape}, v: {v_sol.shape}, rho: {rho_sol.shape}, e: {e_sol.shape}")

# Basic visualization (you can expand this)
plt.figure(figsize=(12, 8))

# Plot x positions vs density
plt.subplot(2, 2, 1)
plt.scatter(x_sol[:, 0], rho_sol, s=5, alpha=0.7)
plt.xlabel('X Position')
plt.ylabel('Density')
plt.title('Density Distribution')

# Plot x positions vs pressure
plt.subplot(2, 2, 2)
plt.scatter(x_sol[:, 0], p_sol, s=5, alpha=0.7, color='red')
plt.xlabel('X Position')
plt.ylabel('Pressure')
plt.title('Pressure Distribution')

# Plot x positions vs velocity x-component
plt.subplot(2, 2, 3)
plt.scatter(x_sol[:, 0], v_sol[:, 0], s=5, alpha=0.7, color='green')
plt.xlabel('X Position')
plt.ylabel('Velocity X')
plt.title('Velocity Distribution')

# Plot x positions vs internal energy
plt.subplot(2, 2, 4)
plt.scatter(x_sol[:, 0], e_sol, s=5, alpha=0.7, color='purple')
plt.xlabel('X Position')
plt.ylabel('Internal Energy')
plt.title('Energy Distribution')

plt.tight_layout()
plt.savefig('sph_solution.png', dpi=150)
plt.show()