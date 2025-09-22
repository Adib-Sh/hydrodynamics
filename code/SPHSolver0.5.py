import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

class SPHSolver:
    def __init__(self, X_0, gamma=1.4, alpha=1.0, beta=1.0, eta=1.2, dt=0.005, n_steps=40):
        self.pos = X_0[:, 0]
        self.v = X_0[:, 3]
        self.m = X_0[:, 6]
        self.rho = X_0[:, 7]
        self.e = X_0[:, 9]
    
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.dt = dt
        self.n_steps = n_steps
        self.N = len(self.pos)
        
        # Set smoothing length
        dx = 0.001875  # Particle spacing
        self.h = 0.02  # Optimized value
        print(f"Using smoothing length h = {self.h}")
         
        self.X_0_flat = np.concatenate([self.pos, self.v, self.rho, self.e])
    
    def c_kernel(self, r, h):
        """Cubic spline kernel function"""
        alpha = 1.0 / h
        R = np.abs(r) / h
        W = np.zeros_like(R)
        
        m1 = (R >= 0) & (R < 1)
        m2 = (R >= 1) & (R < 2)
        
        W[m1] = alpha * (2/3 - R[m1]**2 + 0.5 * R[m1]**3)
        W[m2] = alpha * (1/6) * (2 - R[m2])**3
        
        return W

    def c_kernel_dr(self, r, h):
        """Derivative of cubic spline kernel function"""
        alpha = 1.0 / h**2
        R = np.abs(r) / h
        r_safe = np.where(np.abs(r) < 1e-10, 1e-10, r)
        signR = np.sign(r_safe)
        dW_dr = np.zeros_like(R)
        
        m1 = (R >= 0) & (R < 1)
        m2 = (R >= 1) & (R < 2)
    
        dW_dr[m1] = alpha * (-2*R[m1] + 1.5*R[m1]**2) * signR[m1] / h
        dW_dr[m2] = -alpha * 0.5*(2-R[m2])**2 * signR[m2]
    
        return dW_dr

    def f(self, t, y):
        """SPH equations - main computation function"""
        N = self.N
        
        # Progress monitoring
        if not hasattr(self, 'call_count'):
            self.call_count = 0
            self.last_print_time = 0
        
        self.call_count += 1
        
        # Print progress every 100 calls or every 0.01 time units
        if self.call_count % 100 == 0 or (t - self.last_print_time) > 0.01:
            progress = t / (self.n_steps * self.dt) * 100
            print(f"t={t:.6f} ({progress:.1f}%), call #{self.call_count}")
            self.last_print_time = t
        
        # Extract current state
        pos = y[0*N:1*N]
        v   = y[1*N:2*N]
        rho = y[2*N:3*N]
        e   = y[3*N:4*N]
        
        # Apply safety floors
        rho = np.maximum(rho, 0.01)
        e = np.maximum(e, 0.1)

        # Equation of state
        p = (self.gamma - 1.0) * rho * e
        c = np.sqrt((self.gamma - 1.0) * e)

        # Distance calculations
        xi = pos[:, None]
        xj = pos[None, :]
        xij = xi - xj
        r = np.abs(xij)
    
        # Kernel calculations
        W = self.c_kernel(r, self.h)
        dW_dx = self.c_kernel_dr(xij, self.h)
        
        # Remove self-interactions
        np.fill_diagonal(W, 0)
        np.fill_diagonal(dW_dx, 0)
        
        # Broadcasting arrays
        m_j = self.m[None, :]
        rho_i = rho[:, None]
        rho_j = rho[None, :]
        p_i = p[:, None]
        p_j = p[None, :]
        v_i = v[:, None]
        v_j = v[None, :]
        vij = v_i - v_j
        c_avg = 0.5 * (c[:, None] + c[None, :])
        rho_avg = 0.5 * (rho_i + rho_j)
    
        # Artificial viscosity (Monaghan)
        eps = 0.1 * self.h
        vdotr = vij * xij
        phi_ij = (self.h * vdotr) / (r**2 + eps**2)
        
        Pi = np.zeros_like(phi_ij)
        approaching = vdotr < 0
        Pi[approaching] = ((-self.alpha * c_avg[approaching] * phi_ij[approaching] + 
                           self.beta * phi_ij[approaching]**2) / rho_avg[approaching])
    
        # SPH equations
        drho_dt = np.sum(m_j * vij * dW_dx, axis=1)
        
        pressure_term = (p_i / (rho_i**2)) + (p_j / (rho_j**2)) + Pi
        dv_dt = -np.sum(m_j * pressure_term * dW_dx, axis=1)
        
        de_dt = 0.5 * np.sum(m_j * pressure_term * vij * dW_dx, axis=1)
        
        dpos_dt = v
        
        # Apply stability limits
        max_accel = 100
        dv_dt = np.clip(dv_dt, -max_accel, max_accel)
        drho_dt = np.clip(drho_dt, -1000, 1000)
        de_dt = np.clip(de_dt, -1e6, 1e6)
    
        # Pack derivatives
        dy = np.zeros(4 * N)
        dy[0*N:1*N] = dpos_dt
        dy[1*N:2*N] = dv_dt
        dy[2*N:3*N] = drho_dt
        dy[3*N:4*N] = de_dt
        
        return dy

    def get_max_dt(self, pos, v, rho, e, h):
        """Calculate maximum stable time step using CFL condition"""
        c = np.sqrt((self.gamma - 1.0) * e)
        max_speed = np.max(np.abs(v))
        max_sound = np.max(c)
        dt_cfl = 0.3 * h / (max_speed + max_sound + 1e-10)
        return dt_cfl
    
    def solve(self, method='DOP853'):
        """Solve the SPH system using scipy's solve_ivp"""
        print(f"Starting SPH simulation: {self.N} particles")
        print(f"Time span: 0 to {self.n_steps * self.dt:.6f}")
        
        # Check CFL condition
        max_dt = self.get_max_dt(self.pos, self.v, self.rho, self.e, self.h)
        if self.dt > max_dt:
            self.dt = max_dt * 0.5
            print(f"Reduced dt to {self.dt:.8f} for CFL stability")
        
        t_span = (0, self.n_steps * self.dt)
        t_eval = np.linspace(0, self.n_steps * self.dt, self.n_steps + 1)
        
        self.result = solve_ivp(
            self.f, t_span, self.X_0_flat,
            method=method,
            t_eval=t_eval,
            max_step=self.dt,
            rtol=1e-6,
            atol=1e-8,
            first_step=self.dt/100
        )
        
        if self.result.success:
            print(f"Integration completed successfully!")
            print(f"Total function evaluations: {self.result.nfev}")
            print(f"Final time: {self.result.t[-1]:.6f}")
        else:
            print(f"Integration failed: {self.result.message}")
        
        return self.result
    
    def plot_results(self):
        """Plot the four required quantities: pressure, density, energy, velocity"""
        if not hasattr(self, 'result') or not self.result.success:
            print("No successful results to plot.")
            return
        
        N = self.N
        final_state = self.result.y[:, -1]
        
        pos = final_state[0*N:1*N]
        v = final_state[1*N:2*N]
        rho = final_state[2*N:3*N]
        e = final_state[3*N:4*N]
        p = (self.gamma - 1.0) * rho * e
        
        # Sort by position for plotting
        sort_idx = np.argsort(pos)
        pos_sorted = pos[sort_idx]
        v_sorted = v[sort_idx]
        rho_sorted = rho[sort_idx]
        p_sorted = p[sort_idx]
        e_sorted = e[sort_idx]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'SPH Sod Shock Tube Results (t={self.result.t[-1]:.4f}s)', fontsize=16)
        
        axes[0, 0].plot(pos_sorted, p_sorted, 'b.-', markersize=2)
        axes[0, 0].set_xlabel('x (m)')
        axes[0, 0].set_ylabel('Pressure (Pa)')
        axes[0, 0].set_title('Pressure vs Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(pos_sorted, rho_sorted, 'r.-', markersize=2)
        axes[0, 1].set_xlabel('x (m)')
        axes[0, 1].set_ylabel('Density (kg/m³)')
        axes[0, 1].set_title('Density vs Position')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(pos_sorted, e_sorted, 'g.-', markersize=2)
        axes[1, 0].set_xlabel('x (m)')
        axes[1, 0].set_ylabel('Internal Energy')
        axes[1, 0].set_title('Internal Energy vs Position')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(pos_sorted, v_sorted, 'm.-', markersize=2)
        axes[1, 1].set_xlabel('x (m)')
        axes[1, 1].set_ylabel('Velocity (m/s)')
        axes[1, 1].set_title('Velocity vs Position')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Sod shock tube initial conditions
    m = 0.001875
    N_hi = 320
    N_lo = 80
    N = N_hi + N_lo

    # Left state (high pressure/density)
    p_hi, rho_hi, v_hi, e_hi = 1.0, 1.0, 0.0, 2.5
    
    # Right state (low pressure/density)
    p_lo, rho_lo, v_lo, e_lo = 0.1795, 0.25, 0.0, 1.795

    # Create particle positions (avoid overlap at x=0)
    x_hi = np.linspace(-0.6, -0.001875, num=N_hi)
    x_lo = np.linspace(0.001875, 0.6, num=N_lo)
    x_0 = np.concatenate([x_hi, x_lo])
    
    # Initial conditions
    p_0 = np.concatenate([np.full(N_hi, p_hi), np.full(N_lo, p_lo)])
    rho_0 = np.concatenate([np.full(N_hi, rho_hi), np.full(N_lo, rho_lo)])
    v_0 = np.concatenate([np.full(N_hi, v_hi), np.full(N_lo, v_lo)])
    e_0 = np.concatenate([np.full(N_hi, e_hi), np.full(N_lo, e_lo)])
    
    # Add small velocity perturbation to break initial symmetry
    np.random.seed(42)
    v_perturbation = np.random.normal(0, 1e-6, N)
    v_0 += v_perturbation
    
    print(f"Setup: {N} particles ({N_hi} high-pressure, {N_lo} low-pressure)")
    print(f"Added velocity perturbation: [{np.min(v_perturbation):.2e}, {np.max(v_perturbation):.2e}]")

    # Create initial state vector
    X_0 = np.zeros((N, 10))
    X_0[:, 0] = x_0            # position
    X_0[:, 3] = v_0            # velocity  
    X_0[:, 6] = np.full(N, m)  # mass
    X_0[:, 7] = rho_0          # density
    X_0[:, 8] = p_0            # pressure
    X_0[:, 9] = e_0            # internal energy

    # Create and run solver
    solver = SPHSolver(X_0, dt=1e-7, n_steps=1000)
    result = solver.solve(method='DOP853')
    
    # Plot results if successful
    if result.success:
        solver.plot_results()
    else:
        print("Simulation failed - no plots generated")