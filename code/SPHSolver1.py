# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 02:25:31 2025

@author: adisha
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation

class SPHSolver:
    """
    SPH solver implementation with broadcasting for better performance
    """
    
    def __init__(self, particles, gamma=1.4, alpha=1.0, beta=1.0, eta=1.2):
        """
        Initialize SPH solver with particle data
        """
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
        
        # Calculate constant smoothing length
        rho_avg = np.mean(self.rho)
        m_avg = np.mean(self.m)
        self.h = self.eta * (m_avg / rho_avg)**(1.0)  # 1D
        
        print(f"SPH Solver initialized:")
        print(f"  Particles: {self.N}")
        print(f"  Smoothing length h: {self.h:.6f}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Artificial viscosity: α={self.alpha}, β={self.beta}")
        
        # Pack initial state for integrator
        self.y0 = self._pack_state()
        
        # Progress tracking
        self.start_time = None
        self.call_count = 0
        
    def _pack_state(self):
        """Pack particle state into 1D array for integrator"""
        return np.concatenate([self.x, self.v, self.rho, self.e])
    
    def _unpack_state(self, y):
        """Unpack 1D state array into particle quantities"""
        N = self.N
        x = y[0*N:1*N]
        v = y[1*N:2*N] 
        rho = y[2*N:3*N]
        e = y[3*N:4*N]
        return x, v, rho, e
        
    def kernel_cubic_spline(self, r, h):
        """
        Cubic spline kernel from lecture notes.
        W(R,h) where R = |r|/h, α_d = 1/h for 1D
        """
        R = np.abs(r) / h
        alpha_d = 1.0 / h  # 1D normalization
        
        W = np.zeros_like(R)
        
        # Condition 1: 0 ≤ R < 1
        mask1 = (R >= 0) & (R < 1)
        W[mask1] = alpha_d * (2/3 - R[mask1]**2 + 0.5 * R[mask1]**3)
        
        # Condition 2: 1 ≤ R < 2  
        mask2 = (R >= 1) & (R < 2)
        W[mask2] = alpha_d * (1/6) * (2 - R[mask2])**3
        
        return W
    
    def kernel_gradient(self, r, h):
        """
        Gradient of cubic spline kernel from lecture notes.
        Returns dW/dx
        """
        R = np.abs(r) / h
        alpha_d = 1.0 / h
        
        dW_dx = np.zeros_like(r)
        
        # Condition 1: 0 ≤ R < 1
        mask1 = (R >= 0) & (R < 1)
        dW_dx[mask1] = alpha_d * (-2 + 1.5 * R[mask1]) / h
        
        # Condition 2: 1 ≤ R < 2
        mask2 = (R >= 1) & (R < 2)
        dW_dx[mask2] = -alpha_d * 0.5 * (2 - R[mask2])**2 / (h * R[mask2] + 1e-12)
        
        # Apply sign and handle zero distance
        dW_dx *= np.sign(r)
        dW_dx[np.abs(r) < 1e-12] = 0
        
        return dW_dx
        
    def equation_of_state(self, rho, e):
        """Ideal gas EOS: p = (γ-1)ρe"""
        return (self.gamma - 1.0) * rho * e
        
    def sound_speed(self, e):
        """Speed of sound: c = √((γ-1)e)"""
        return np.sqrt((self.gamma - 1.0) * np.maximum(e, 1e-10))
        
    def artificial_viscosity_broadcast(self, v_i, v_j, x_i, x_j, rho_i, rho_j, c_i, c_j, h_ij):
        """
        Artificial viscosity using broadcasting - FIXED VERSION
        """
        v_ij = v_i - v_j  # N x N matrix
        x_ij = x_i - x_j  # N x N matrix
        r_ij = np.abs(x_ij)  # N x N matrix
        
        # Create mask for approaching particles (v_ij * x_ij < 0)
        approaching_mask = (v_ij * x_ij) < 0
        
        # Calculate φ_ij only for approaching particles
        phi_ij = np.zeros_like(x_ij)
        
        # Use the mask to calculate only for approaching particles
        phi_ij = np.where(approaching_mask, 
                         (h_ij * v_ij * x_ij) / (r_ij**2 + (0.1 * h_ij)**2),
                         0.0)
        
        # Average quantities
        rho_avg = 0.5 * (rho_i + rho_j)
        c_avg = 0.5 * (c_i + c_j)
        
        # Monaghan viscosity - only apply for approaching particles
        Pi_ij = np.where(approaching_mask,
                        (-self.alpha * c_avg * phi_ij + self.beta * phi_ij**2) / rho_avg,
                        0.0)
        
        return Pi_ij
    
    def sph_equations_broadcast(self, t, y):
        """
        SPH equations using broadcasting for better performance
        """
        self.call_count += 1
        
        # Progress monitoring
        if self.call_count % 100 == 0:
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                print(f"t={t:.6f}, calls={self.call_count:,}, elapsed={elapsed:.1f}s")
        
        # Unpack current state
        x, v, rho, e = self._unpack_state(y)
        
        # Calculate pressure and sound speed
        p = self.equation_of_state(rho, e)
        c = self.sound_speed(e)
        
        # Create distance matrix using broadcasting
        x_matrix = x[:, np.newaxis] - x[np.newaxis, :]  # N x N matrix of x_i - x_j
        r_matrix = np.abs(x_matrix)  # N x N matrix of distances
        
        # Create kernel support mask
        support_mask = (r_matrix <= 2.0 * self.h) & (r_matrix > 1e-12)
        
        # Calculate kernel and gradient for all pairs
        W_matrix = np.zeros_like(r_matrix)
        grad_W_matrix = np.zeros_like(x_matrix)
        
        # Apply kernel functions only to supported pairs
        W_matrix[support_mask] = self.kernel_cubic_spline(r_matrix[support_mask], self.h)
        grad_W_matrix[support_mask] = self.kernel_gradient(x_matrix[support_mask], self.h)
        
        # Set diagonal to zero (no self-interaction)
        np.fill_diagonal(W_matrix, 0)
        np.fill_diagonal(grad_W_matrix, 0)
        
        # Calculate density using summation method (broadcasted)
        rho_new = np.sum(self.m * W_matrix, axis=1)
        rho = np.maximum(rho_new, 0.01)
        
        # Recalculate pressure with new density
        p = self.equation_of_state(rho, e)
        c = self.sound_speed(e)
        
        # Prepare arrays for broadcasting
        m_matrix = self.m[np.newaxis, :]  # 1 x N mass matrix
        rho_i_matrix = rho[:, np.newaxis]  # N x 1 density matrix
        rho_j_matrix = rho[np.newaxis, :]  # 1 x N density matrix
        p_i_matrix = p[:, np.newaxis]  # N x 1 pressure matrix
        p_j_matrix = p[np.newaxis, :]  # 1 x N pressure matrix
        c_i_matrix = c[:, np.newaxis]  # N x 1 sound speed matrix
        c_j_matrix = c[np.newaxis, :]  # 1 x N sound speed matrix
        v_matrix = v[:, np.newaxis] - v[np.newaxis, :]  # N x N velocity difference matrix
        
        # Artificial viscosity (broadcasted)
        h_ij_matrix = np.full_like(x_matrix, self.h)
        Pi_ij_matrix = self.artificial_viscosity_broadcast(
            v[:, np.newaxis], v[np.newaxis, :], 
            x[:, np.newaxis], x[np.newaxis, :],
            rho_i_matrix, rho_j_matrix,
            c_i_matrix, c_j_matrix,
            h_ij_matrix
        )
        
        # Pressure term (broadcasted)
        pressure_term_matrix = (p_i_matrix / rho_i_matrix**2 + 
                               p_j_matrix / rho_j_matrix**2 + 
                               Pi_ij_matrix)
        
        # Momentum equation (sum over j)
        dv_dt = -np.sum(m_matrix * pressure_term_matrix * grad_W_matrix, axis=1)
        
        # Energy equation (sum over j)
        de_dt = 0.5 * np.sum(m_matrix * pressure_term_matrix * v_matrix * grad_W_matrix, axis=1)
        
        # Position equation
        dx_dt = v.copy()
        
        # Density equation (continuity) - optional, we're using summation density
        drho_dt = np.zeros_like(rho)
        
        # Apply derivative limits for stability
        max_accel = 1e3
        dv_dt = np.clip(dv_dt, -max_accel, max_accel)
        de_dt = np.clip(de_dt, -max_accel, max_accel)
        
        # Pack derivatives
        dy_dt = np.concatenate([dx_dt, dv_dt, drho_dt, de_dt])
        
        return dy_dt
    
    def solve(self, t_final=0.2, dt_max=1e-5, method='DOP853'):
        """
        Solve SPH system using adaptive time stepping
        """
        print(f"\nStarting SPH integration:")
        print(f"  Final time: {t_final}")
        print(f"  Max time step: {dt_max}")
        print(f"  Method: {method}")
        
        self.start_time = time.time()
        self.call_count = 0
        
        # Time span and evaluation points  
        t_span = (0, t_final)
        t_eval = np.linspace(0, t_final, 41)  # 40 time steps as in lab manual
        
        # Solve using scipy
        sol = solve_ivp(
            self.sph_equations_broadcast,
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
        """Plot the standard Sod shock tube results"""
        if not sol.success:
            print("Cannot plot - integration failed")
            return
            
        # Final state
        final_state = sol.y[:, -1]
        x, v, rho, e = self._unpack_state(final_state)
        p = self.equation_of_state(rho, e)
        
        # Sort by position
        idx = np.argsort(x)
        x_sort = x[idx]
        v_sort = v[idx]  
        rho_sort = rho[idx]
        p_sort = p[idx]
        e_sort = e[idx]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Sod Shock Tube (SPH) - t = {sol.t[-1]:.3f}s", fontsize=14)
        
        # Pressure
        axes[0,0].plot(x_sort, p_sort, 'b-o', markersize=2, label='SPH')
        axes[0,0].set_xlabel('Position x (m)')
        axes[0,0].set_ylabel('Pressure p (Pa)')
        axes[0,0].set_title('Pressure')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Density
        axes[0,1].plot(x_sort, rho_sort, 'r-o', markersize=2, label='SPH')
        axes[0,1].set_xlabel('Position x (m)')
        axes[0,1].set_ylabel('Density ρ (kg/m³)')
        axes[0,1].set_title('Density')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Velocity  
        axes[1,0].plot(x_sort, v_sort, 'm-o', markersize=2, label='SPH')
        axes[1,0].set_xlabel('Position x (m)')
        axes[1,0].set_ylabel('Velocity v (m/s)')
        axes[1,0].set_title('Velocity')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Internal energy
        axes[1,1].plot(x_sort, e_sort, 'g-o', markersize=2, label='SPH')
        axes[1,1].set_xlabel('Position x (m)')
        axes[1,1].set_ylabel('Internal Energy e')
        axes[1,1].set_title('Internal Energy')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        

def setup_sod_shock_tube():
    """
    Set up initial conditions for Sod's shock tube problem
    """
    print("Setting up Sod's shock tube initial conditions...")
    
    # Parameters from Table 1 in lab manual
    N_left = 320   # particles in high density region
    N_right = 80   # particles in low density region  
    N_total = N_left + N_right
    
    m = 0.001875   # particle mass (kg)
    
    # Left state (x ≤ 0): high pressure/density
    rho_L, v_L, e_L, p_L = 1.0, 0.0, 2.5, 1.0
    dx_L = 0.001875  # particle spacing
    
    # Right state (x > 0): low pressure/density  
    rho_R, v_R, e_R, p_R = 0.25, 0.0, 1.795, 0.1795
    dx_R = 0.0075    # particle spacing
    
    # Create particle positions
    x_left = np.linspace(-0.6, -dx_L, N_left)
    x_right = np.linspace(dx_R, 0.6, N_right)
    
    # Combine positions
    x = np.concatenate([x_left, x_right])
    
    # Initial velocities
    v = np.zeros(N_total)
    
    # Densities
    rho = np.concatenate([np.full(N_left, rho_L), np.full(N_right, rho_R)])
    
    # Internal energies
    e = np.concatenate([np.full(N_left, e_L), np.full(N_right, e_R)])
    
    # Masses (all same)
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
    # Set up Sod shock tube
    particles = setup_sod_shock_tube()
    
    # Create SPH solver
    solver = SPHSolver(particles, gamma=1.4, alpha=1.0, beta=1.0, eta=1.2)
    
    # Solve with smaller time step
    solution = solver.solve(t_final=0.2, dt_max=0.0005, method='RK45')
    
    # Plot results
    if solution.success:
        solver.plot_solution(solution)

    
    
    print("\nSod shock tube simulation complete!")
    
def create_shock_tube_animation(sol):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    def update(frame):
        for ax in axes.flat:
            ax.clear()
        
        # Plot current state
        state = sol.y[:, frame]
        x, v, rho, e = solver._unpack_state(state)
        p = solver.equation_of_state(rho, e)
        
        # Plot with exact solution comparison
        axes[0,0].plot(x, p, 'bo', markersize=2)
        axes[0,0].set_title(f'Pressure, t={sol.t[frame]:.3f}s')
        axes[0,1].plot(x, v, 'bo', markersize=2)
        axes[0,1].set_title(f'Velocity, t={sol.t[frame]:.3f}s')
        axes[1,0].plot(x, rho, 'bo', markersize=2)
        axes[1,0].set_title(f'Density, t={sol.t[frame]:.3f}s')
        axes[1,1].plot(x, e, 'bo', markersize=2)
        axes[1,1].set_title(f'energy, t={sol.t[frame]:.3f}s')
        # ... repeat for other variables
        
    ani = FuncAnimation(fig, update, frames=len(sol.t), interval=100)
    ani.save('shock_tube_evolution.gif', writer='ffmpeg')
    print("saved")
create_shock_tube_animation(solution)