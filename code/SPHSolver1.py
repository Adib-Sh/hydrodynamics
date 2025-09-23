import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation

class SPHSolver:
    
    def __init__(self, particles, gamma=1.4, alpha=1.0, beta=1.0, eta=1.2):

        self.x = particles['x'].copy()
        self.v = particles['v'].copy()
        self.rho = particles['rho'].copy()
        self.e = particles['e'].copy()
        self.m = particles['m'].copy()
        
        self.N = len(self.x)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        
        # smoothing length
        rho_avg = np.mean(self.rho)
        m_avg = np.mean(self.m)
        self.h = 0.001 #self.eta * (m_avg / rho_avg)**(1.0)  # 1D
        
        print(f"  Particles: {self.N}")
        print(f"  Smoothing length h: {self.h:.6f}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Artificial viscosity: alpha={self.alpha}, beta={self.beta}")
        
        # Pack initial state for integrator
        self.y0 = self._pack_state()
        
        # Progress tracking
        self.start_time = None
        self.call_count = 0
        
    def _pack_state(self):
        return np.concatenate([self.x, self.v, self.rho, self.e])
    
    def _unpack_state(self, y):
        N = self.N
        x = y[0*N:1*N]
        v = y[1*N:2*N] 
        rho = y[2*N:3*N]
        e = y[3*N:4*N]
        return x, v, rho, e
        
    def kernel_cubic_spline(self, dx, h):
        R = np.abs(dx) / h
        alpha_d = 1.0 / h
        
        W = np.zeros_like(R)
        
        # Condition 0 ≤ R < 1
        mask1 = (R >= 0) & (R < 1)
        W[mask1] = alpha_d * (2/3 - R[mask1]**2 + 0.5 * R[mask1]**3)
        
        # Condition 1 ≤ R < 2  
        mask2 = (R >= 1) & (R < 2)
        W[mask2] = alpha_d * (1/6) * (2 - R[mask2])**3
        
        return W
    
    def kernel_gradient(self, dx, h):
         r = np.abs(dx)
         R = r / h
         alpha_d = 1.0 / h
         dW_dx = np.zeros_like(dx)
    
         # 0 ≤ R < 1
         mask1 = (R >= 0) & (R < 1)
         dW_dx[mask1] = alpha_d * (-2 + 1.5 * R[mask1]) * r[mask1] / h**2
    
         # 1 ≤ R < 2
         mask2 = (R >= 1) & (R < 2)
         dW_dx[mask2] = -0.5 * alpha_d * (2 - R[mask2])**2 * dx[mask2]/(h*r[mask2])
    
         dW_dx *= np.sign(dx)
    
         return dW_dx
        
    def p(self, rho, e):
        return (self.gamma - 1.0) * rho * e
        
    def sound_speed(self, e):
        return np.sqrt((self.gamma - 1.0) * np.maximum(e, 1e-10))
        
    def artificial_viscosity(self, v_matrix, dx_matrix, rho_i, rho_j, c_i, c_j, h_ij):
        v_ij = v_matrix
        x_ij = dx_matrix
        r_ij = np.abs(x_ij)
        
        # mask for approaching particles (v_ij * x_ij < 0)
        mask = (v_ij * x_ij) < 0
        
        phi_ij = np.zeros_like(x_ij)
        phi_ij = np.where(mask, 
                         (h_ij * v_ij * x_ij) / (r_ij**2 + (0.1 * h_ij)**2),
                         0.0)
        
        rho_avg = 0.5 * (rho_i + rho_j)
        c_avg = 0.5 * (c_i + c_j)
        
        # viscosity - only for approaching particles
        Pi_ij = np.where(mask,
                        (-self.alpha * c_avg * phi_ij + self.beta * phi_ij**2) / rho_avg,
                        0.0)
        
        return Pi_ij
    
    def sph(self, t, y):

        self.call_count += 1
        if self.call_count % 100 == 0:
            if self.start_time is not None:
                print(f"t={t:.6f}, calls={self.call_count:}")

        x, v, rho, e = self._unpack_state(y)
        p = self.p(rho, e)
        c = self.sound_speed(e)
        dx_matrix = x[:, np.newaxis] - x[np.newaxis, :]  # N x N matrix of x_i - x_j
        r_matrix = np.abs(dx_matrix)  # N x N matrix of distances
        
        
        W_matrix = np.zeros_like(dx_matrix)
        grad_W_matrix = np.zeros_like(dx_matrix)
        W_matrix = self.kernel_cubic_spline(dx_matrix, self.h)
        grad_W_matrix = self.kernel_gradient(dx_matrix, self.h)
        
        rho_new = np.sum(self.m * W_matrix, axis=1)
        p = self.p(rho_new, e)
        c = self.sound_speed(e)
        
        m_matrix = self.m[np.newaxis, :]  # 1 x N mass matrix
        rho_i_matrix = rho_new[:, np.newaxis]  # N x 1 density matrix
        rho_j_matrix = rho_new[np.newaxis, :]  # 1 x N density matrix
        p_i_matrix = p[:, np.newaxis]  # N x 1 pressure matrix
        p_j_matrix = p[np.newaxis, :]  # 1 x N pressure matrix
        c_i_matrix = c[:, np.newaxis]  # N x 1 sound speed matrix
        c_j_matrix = c[np.newaxis, :]  # 1 x N sound speed matrix
        v_matrix = v[:, np.newaxis] - v[np.newaxis, :]  # N x N velocity difference matrix
        h_ij_matrix = np.full_like(dx_matrix, self.h)
        Pi_ij_matrix = self.artificial_viscosity(
            v_matrix, 
            dx_matrix,
            rho_i_matrix, rho_j_matrix,
            c_i_matrix, c_j_matrix,
            h_ij_matrix
        )
        Pi_ij_matrix = self.artificial_viscosity(
            v_matrix, 
            dx_matrix,
            rho_i_matrix, rho_j_matrix,
            c_i_matrix, c_j_matrix,
            h_ij_matrix
        )
        pressure_term_matrix = (p_i_matrix / rho_i_matrix**2 + 
                               p_j_matrix / rho_j_matrix**2)
                               #+ Pi_ij_matrix)
        
        # Acceleration
        dv_dt = -np.sum(m_matrix * pressure_term_matrix * grad_W_matrix, axis=1)
        
        # Energy 
        de_dt = 0.5 * np.sum(m_matrix * pressure_term_matrix * v_matrix * grad_W_matrix, axis=1)
        
        # Position
        dx_dt = v.copy()
        
        # Density
        drho_dt = np.zeros_like(rho_new)
        dy_dt = np.concatenate([dx_dt, dv_dt, drho_dt, de_dt])
        return dy_dt
    
    def solve(self, t_final=0.2, dt_max=1e-5, method='RK45'):

        print(f"\nStarting SPH integration:")
        print(f"  Final time: {t_final}")
        print(f"  Max time step: {dt_max}")
        print(f"  Method: {method}")
        
        self.start_time = time.time()
        self.call_count = 0
        
        t_span = (0, t_final)
        t_eval = np.linspace(0, t_final, 41)
        
        # Solve using scipy
        sol = solve_ivp(
            self.sph,
            t_span, 
            self.y0,
            method=method,
            t_eval=t_eval,
            max_step=dt_max,
            rtol=1e-6,
            atol=1e-8,
            dense_output=False
        )
        
        elapsed = time.time() - self.start_time
        
        if sol.success:
            print(f"\nIntegration successful!")
            print(f"  Function evaluations: {sol.nfev:,}")
            print(f"  Time steps: {len(sol.t)}")
            print(f"  Wall time: {elapsed:.2f}s")
        else:
            print(f"\nIntegration failed: {sol.message}")
            
        return sol
    
    def plot_solution(self, sol):

        final_state = sol.y[:, -1]
        x, v, rho, e = self._unpack_state(final_state)
        p = self.p(rho, e)
        
        idx = np.argsort(x)
        x_sort = x[idx]
        v_sort = v[idx]  
        rho_sort = rho[idx]
        p_sort = p[idx]
        e_sort = e[idx]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Sod Shock Tube (SPH) - t = {sol.t[-1]:.3f}s", fontsize=14)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Sod Shock Tube (SPH) - t = {sol.t[-1]:.3f}s", fontsize=14)
        
        # Pressure
        axes[0,0].scatter(x_sort, p_sort, s=10, c='b', marker='o', label='SPH')
        axes[0,0].set_xlabel('Position x (m)')
        axes[0,0].set_ylabel('Pressure p (Pa)')
        axes[0,0].set_title('Pressure')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Density
        axes[0,1].scatter(x_sort, rho_sort, s=10, c='r', marker='o', label='SPH')
        axes[0,1].set_xlabel('Position x (m)')
        axes[0,1].set_ylabel('Density ρ (kg/m³)')
        axes[0,1].set_title('Density')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Velocity  
        axes[1,0].scatter(x_sort, v_sort, s=10, c='m', marker='o', label='SPH')
        axes[1,0].set_xlabel('Position x (m)')
        axes[1,0].set_ylabel('Velocity v (m/s)')
        axes[1,0].set_title('Velocity')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Internal energy
        axes[1,1].scatter(x_sort, e_sort, s=10, c='g', marker='o', label='SPH')
        axes[1,1].set_xlabel('Position x (m)')
        axes[1,1].set_ylabel('Internal Energy e')
        axes[1,1].set_title('Internal Energy')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        

def setup_sod_shock_tube():
    
    N_left = 320
    N_right = 80
    N_total = N_left + N_right
    
    m = 0.001875
    
    rho_L, v_L, e_L, p_L = 1.0, 0.0, 2.5, 1.0
    dx_L = 0.001875
    
    rho_R, v_R, e_R, p_R = 0.25, 0.0, 1.795, 0.1795
    dx_R = 0.0075
    
    x_left = np.linspace(-0.6, -dx_L, N_left)
    x_right = np.linspace(dx_R, 0.6, N_right)
    x = np.concatenate([x_left, x_right])
    
    v = np.zeros(N_total)
    
    rho = np.concatenate([np.full(N_left, rho_L), np.full(N_right, rho_R)])
    
    e = np.concatenate([np.full(N_left, e_L), np.full(N_right, e_R)])
    
    masses = np.full(N_total, m)
    
    particles = {
        'x': x,
        'v': v, 
        'rho': rho,
        'e': e,
        'm': masses
    }
    
    print(f"Created {N_total} particles:")
    print(f"  Left region: {N_left} particles, ρ={rho_L}, p={p_L}, e={e_L}")
    print(f"  Right region: {N_right} particles, ρ={rho_R}, p={p_R}, e={e_R}")
    print(f"  Domain: [{np.min(x):.3f}, {np.max(x):.3f}] m")
    
    return particles


if __name__ == "__main__":

    particles = setup_sod_shock_tube()
    
    solver = SPHSolver(particles, gamma=1.4, alpha=1.0, beta=1.0, eta=1.2)
    
    solution = solver.solve(t_final=0.02, dt_max=0.05, method='RK45')
    
    if solution.success:
        solver.plot_solution(solution)

    
    
    print("\nSod shock tube simulation complete!")
    
def create_shock_tube_animation(sol):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    def update(frame):
        for ax in axes.flat:
            ax.clear()

        state = sol.y[:, frame]
        x, v, rho, e = solver._unpack_state(state)
        p = solver.p(rho, e)
        
        axes[0,0].plot(x, p, 'bo', markersize=2)
        axes[0,0].set_title(f'Pressure, t={sol.t[frame]:.3f}s')
        axes[0,1].plot(x, v, 'bo', markersize=2)
        axes[0,1].set_title(f'Velocity, t={sol.t[frame]:.3f}s')
        axes[1,0].plot(x, rho, 'bo', markersize=2)
        axes[1,0].set_title(f'Density, t={sol.t[frame]:.3f}s')
        axes[1,1].plot(x, e, 'bo', markersize=2)
        axes[1,1].set_title(f'energy, t={sol.t[frame]:.3f}s')
        
    ani = FuncAnimation(fig, update, frames=len(sol.t), interval=100)
    ani.save('shock_tube_evolution.gif', writer='ffmpeg')
    print("saved")
create_shock_tube_animation(solution)