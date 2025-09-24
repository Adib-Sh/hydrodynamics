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


def kernel_cubic_spline(dx, h):
    R = np.abs(dx) / h
    alpha_d = 3.0 / (2.0 * np.pi * h**3)  # 3D normalization
    
    W = np.zeros_like(R)
    
    # Condition 0 ≤ R < 1
    mask1 = (R >= 0) & (R < 1)
    W[mask1] = alpha_d * (2/3 - R[mask1]**2 + 0.5 * R[mask1]**3)
    
    # Condition 1 ≤ R < 2  
    mask2 = (R >= 1) & (R < 2)
    W[mask2] = alpha_d * (1/6) * (2 - R[mask2])**3
    
    return W

def kernel_gradient(dx, h):
    r = np.linalg.norm(dx, axis=-1) if dx.ndim > 1 else np.linalg.norm(dx)
    R = r / h
    alpha_d = 3.0 / (2.0 * np.pi * h**3)
    
    dW_dr = np.zeros_like(r)
    
    # Condition 0 ≤ R < 1
    mask1 = (R >= 0) & (R < 1)
    dW_dr[mask1] = alpha_d * (-2*R[mask1] + 1.5*R[mask1]**2) / h
    
    # Condition 1 ≤ R < 2
    mask2 = (R >= 1) & (R < 2)
    dW_dr[mask2] = -alpha_d * 0.5 * (2 - R[mask2])**2 / (h * R[mask2])
    
    return dW_dr



def p(gamma, rho, e):
    return (gamma - 1.0) * rho * e
    
def sound_speed(e):
    return np.sqrt((gamma - 1.0) * np.maximum(e, 1e-10))

def artificial_viscosity(v_matrix, dx_matrix, rho_i, rho_j, c_i, c_j, h_ij):
    v_ij = v_matrix
    x_ij = dx_matrix
    r_ij = np.abs(x_ij)
    
    # mask for v_ij * x_ij < 0
    mask = (v_ij * x_ij) < 0
    
    phi_ij = np.zeros_like(x_ij)
    phi_ij = np.where(mask, 
                     (h_ij * v_ij * x_ij) / (r_ij**2 + (0.1 * h_ij)**2),
                     0.0)
    
    rho_avg = 0.5 * (rho_i + rho_j)
    c_avg = 0.5 * (c_i + c_j)
    c_avg = c_avg[:,:,np.newaxis]
    
    # viscosity
    Pi_ij = np.where(mask,
                    (-alpha * c_avg * phi_ij + beta * phi_ij**2) / rho_avg,
                    0.0)
    
    return Pi_ij




# Number of time steps
dt = 0.005
n_steps = 40
t_span = (0, dt * n_steps)
t_eval = np.linspace(t_span[0], t_span[1], n_steps)


def sph(t, y):
    N = N_total

    # --- Reshape y to match original particle data ---
    x = y[0:3*N].reshape(N, 3)        # positions
    v = y[3*N:6*N].reshape(N, 3)      # velocities
    m = y[6*N:7*N]                     # masses
    rho = y[7*N:8*N]                   # densities
    e = y[8*N:9*N]                     # energy

    h = 1e7

    dx_matrix = x[:, np.newaxis] - x[np.newaxis, :]
    # dx_matrix = x[:, np.newaxis, :] - x[np.newaxis, :, :]   # (N,N,3)
    r = np.linalg.norm(dx_matrix, axis=-1)                    # (N,N)
    W_matrix = kernel_cubic_spline(dx_matrix, h)                     # (N,N)
    grad_W_scalar = kernel_gradient(dx_matrix, h)                    # (N,N)
    
    grad_W_vector = grad_W_scalar[:, :, np.newaxis] 
    
    
    # shape (N,N,3)


    m_new = m[np.newaxis, :, np.newaxis]  # (1,N,1)
    rho_new = np.sum(m_new * W_matrix, axis=1)  # (N,3)? check

    rho = rho_new
    p_matrix = p(gamma, rho_new[:,0], e[:, np.newaxis])  # if needed reshape
    c = sound_speed(e)

    rho_i_matrix = rho[:, np.newaxis]
    rho_j_matrix = rho[np.newaxis, :]
    p_i_matrix = p_matrix
    p_j_matrix = p_matrix[np.newaxis, :]
    c_i_matrix = c[:, np.newaxis]
    c_j_matrix = c[np.newaxis, :]
    
    h_ij_matrix = np.full_like(dx_matrix, h)
    
    # artificial viscosity
    Pi_ij_matrix = artificial_viscosity(
        v, dx_matrix,
        rho_i_matrix, rho_j_matrix,
        c_i_matrix, c_j_matrix,
        h_ij_matrix
    )
    
    pressure_term_matrix = (p_i_matrix / rho_i_matrix**2 + 
                           p_j_matrix / rho_j_matrix**2 + Pi_ij_matrix)
    
    # Acceleration
    dv_dt = -np.sum(m_new * pressure_term_matrix[:, :, np.newaxis] * grad_W_vector, axis=1)

    
    # Energy
    de_dt = 0.5 * np.sum(m_new * pressure_term_matrix * v * grad_W_vector, axis=1)
    
    # Position derivative
    dx_dt = v.copy()
    
    # Pack derivatives
    dy_dt = np.concatenate([
    dx_dt.ravel(),
    dv_dt.ravel(),
    rho_new.ravel(),
    de_dt.ravel()
])

    
    return dy_dt


# Initial state vector
y0 = np.concatenate([
    planet_data['x'].ravel(),
    planet_data['v'].ravel(),
    planet_data['m'].ravel(),
    planet_data['rho'].ravel(),
    planet_data['e'].ravel()
])
sol = solve_ivp(sph, t_span, y0, t_eval=t_eval, method='RK45', max_step=dt/10, rtol=1e-6, atol=1e-8)

# solutions
N = N_total
x = y[0:3*N].reshape(N, 3)
v = y[3*N:6*N].reshape(N, 3)
m = y[6*N:7*N]        # shape (N,)
rho = y[7*N:8*N]      # shape (N,)
e = y[8*N:9*N]        # shape (N,)

p_sol = np.zeros_like(rho_sol)
for i in range(n_steps):
    p_sol[:, i] = p(gamma, rho_sol[:, i], e_sol[:, i])
'''
def unpack_state(y):
    N = 200
    x = y[0*N:1*N]
    v = y[1*N:2*N] 
    rho = y[2*N:3*N]
    e = y[3*N:4*N]
    p_calc = p(gamma, rho, e)  # Add this line
    return x, v, rho, e, p_calc

def plot_solution(sol):
    final_state = sol.y[:, -1]
    x, v, rho, e, p_calc = unpack_state(final_state)  # Add p_calc
    
    idx = np.argsort(x)
    x_sort = x[idx]
    v_sort = v[idx]  
    rho_sort = rho[idx]
    p_sort = p_calc[idx]
    e_sort = e[idx]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Sod Shock Tube (SPH) - t = {sol.t[-1]:.3f}s", fontsize=14)
    
    # Pressure
    axes[0,0].scatter(x_sort, p_sort, s=10, c='b', marker='o', label='SPH')
    axes[0,0].set_xlabel('Position x (m)')
    axes[0,0].set_ylabel('Pressure p (Pa)')
    axes[0,0].set_title('Pressure')
    axes[0,0].set_xlim(-0.4, 0.4)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Density
    axes[0,1].scatter(x_sort, rho_sort, s=10, c='r', marker='o', label='SPH')
    axes[0,1].set_xlabel('Position x (m)')
    axes[0,1].set_ylabel('Density ρ (kg/m³)')
    axes[0,1].set_title('Density')
    axes[0,1].set_xlim(-0.4, 0.4)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Velocity  
    axes[1,0].scatter(x_sort, v_sort, s=10, c='m', marker='o', label='SPH')
    axes[1,0].set_xlabel('Position x (m)')
    axes[1,0].set_ylabel('Velocity v (m/s)')
    axes[1,0].set_title('Velocity')
    axes[1,0].set_xlim(-0.4, 0.4)
    axes[1,0].set_ylim(-0.5, 1)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # Internal energy
    axes[1,1].scatter(x_sort, e_sort, s=10, c='g', marker='o', label='SPH')
    axes[1,1].set_xlabel('Position x (m)')
    axes[1,1].set_ylabel('Internal Energy e')
    axes[1,1].set_title('Internal Energy')
    axes[1,1].set_xlim(-0.4, 0.4)
    axes[1,1].set_ylim(1.4, 2.6)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig("Sod Shock Tube (SPH).png")
    plt.show()
    
plot_solution(sol)

def create_shock_tube_animation(sol):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    def update(frame):
        for ax in axes.flat:
            ax.clear()
        
        state = sol.y[:, frame]
        x, v, rho, e, p_calc = unpack_state(state)  # Use your existing unpack_state function
        
        # Sort by position for proper plotting
        idx = np.argsort(x)
        x_sort = x[idx]
        v_sort = v[idx]
        rho_sort = rho[idx]
        p_sort = p_calc[idx]
        e_sort = e[idx]
        
        axes[0,0].scatter(x_sort, p_sort, s=10, c='b', marker='o')
        axes[0,0].set_title(f'Pressure, t={sol.t[frame]:.3f}s')
        axes[0,0].set_xlabel('Position x (m)')
        axes[0,0].set_ylabel('Pressure p (Pa)')
        axes[0,0].set_xlim(-0.4, 0.4)
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].scatter(x_sort, v_sort, s=10, c='m', marker='o')
        axes[0,1].set_title(f'Velocity, t={sol.t[frame]:.3f}s')
        axes[0,1].set_xlabel('Position x (m)')
        axes[0,1].set_ylabel('Velocity v (m/s)')
        axes[0,1].set_xlim(-0.4, 0.4)
        axes[0,1].set_ylim(-0.5, 1)
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].scatter(x_sort, rho_sort, s=10, c='r', marker='o')
        axes[1,0].set_title(f'Density, t={sol.t[frame]:.3f}s')
        axes[1,0].set_xlabel('Position x (m)')
        axes[1,0].set_ylabel('Density ρ (kg/m³)')
        axes[1,0].set_xlim(-0.4, 0.4)
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].scatter(x_sort, e_sort, s=10, c='g', marker='o')
        axes[1,1].set_title(f'Internal Energy, t={sol.t[frame]:.3f}s')
        axes[1,1].set_xlabel('Position x (m)')
        axes[1,1].set_ylabel('Internal Energy e')
        axes[1,1].set_xlim(-0.4, 0.4)
        axes[1,1].set_ylim(1.4, 2.6)
        axes[1,1].grid(True, alpha=0.3)
        
    ani = FuncAnimation(fig, update, frames=len(sol.t), interval=100)
    ani.save('shock_tube_evolution.gif', writer='ffmpeg')
    print("Animation saved as shock_tube_evolution.gif")
    plt.show()

create_shock_tube_animation(sol)
'''