import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm

class SPH3DSolver:
   
    def __init__(self, particles, gamma=1.4, alpha=1.0, beta=1.0, eta=1.2, G=6.67430e-11):
        self.x = particles['x'].copy()  # N x 3 array
        self.v = particles['v'].copy()  # N x 3 array
        self.rho = particles['rho'].copy()
        self.e = particles['e'].copy()
        self.m = particles['m'].copy()
        
        self.N = len(self.x)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.G = G  # gravitational constant
        
        # Calculate smoothing length
        rho_avg = np.mean(self.rho)
        m_avg = np.mean(self.m)
        self.h = self.eta * (m_avg / rho_avg)**(1.0/3.0)  # 3D
        
        print(f"3D SPH Solver initialized:")
        print(f"  Particles: {self.N}")
        print(f"  Smoothing length h: {self.h:.6f}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Gravitational constant G: {self.G}")
        print(f"  Artificial viscosity: α={self.alpha}, β={self.beta}")
        
        # Pack initial state for integrator
        self.y0 = self._pack_state()
        
        # Progress tracking
        self.start_time = None
        self.call_count = 0
        
    def _pack_state(self):
        # Flatten: x (Nx3), v (Nx3), rho (N), e (N)
        return np.concatenate([self.x.flatten(), self.v.flatten(), self.rho, self.e])
    
    def _unpack_state(self, y):
        N = self.N
        x = y[0:3*N].reshape(N, 3)
        v = y[3*N:6*N].reshape(N, 3)
        rho = y[6*N:7*N]
        e = y[7*N:8*N]
        return x, v, rho, e
        
    def kernel_cubic_spline_3d(self, r, h):

        R = np.abs(r) / h
        alpha_d = 3.0 / (2.0 * np.pi * h**3)  # 3D normalization
        
        W = np.zeros_like(R)
        
        # Condition 0 ≤ R < 1
        mask1 = (R >= 0) & (R < 1)
        W[mask1] = alpha_d * (2/3 - R[mask1]**2 + 0.5 * R[mask1]**3)
        
        # Condition 1 ≤ R < 2  
        mask2 = (R >= 1) & (R < 2)
        W[mask2] = alpha_d * (1/6) * (2 - R[mask2])**3
        
        return W
    
    def kernel_gradient_3d(self, r_vec, h):

        r = np.linalg.norm(r_vec, axis=-1) if r_vec.ndim > 1 else np.linalg.norm(r_vec)
        R = r / h
        alpha_d = 3.0 / (2.0 * np.pi * h**3)
        
        dW_dr = np.zeros_like(r)
        
        # Condition 0 ≤ R < 1
        mask1 = (R >= 0) & (R < 1)
        dW_dr[mask1] = alpha_d * (-2*R[mask1] + 1.5*R[mask1]**2) / h
        
        # Condition 1 ≤ R < 2
        mask2 = (R >= 1) & (R < 2)
        dW_dr[mask2] = -alpha_d * 0.5 * (2 - R[mask2])**2 / (h * R[mask2] + 1e-12)
        
        # Convert to gradient vector
        if r_vec.ndim > 1:
            safe_r = np.where(r < 1e-12, 1e-12, r)
            grad_W = dW_dr[..., np.newaxis] * r_vec / safe_r[..., np.newaxis]
            grad_W[r < 1e-12] = 0
        else:
            safe_r = max(r, 1e-12)
            grad_W = dW_dr * r_vec / safe_r if r > 1e-12 else np.zeros_like(r_vec)
        
        return grad_W
        
    def gravitational_potential_gradient(self, r, h):

        R = r / h
        
        dphi_dr = np.zeros_like(r)
        
        # Condition 0 ≤ R < 1
        mask1 = (R >= 0) & (R < 1)
        dphi_dr[mask1] = (1/h**2) * (4/3 * R[mask1] - 6/5 * R[mask1]**3 + 0.5 * R[mask1]**4)
        
        # Condition 1 ≤ R < 2
        mask2 = (R >= 1) & (R < 2)
        dphi_dr[mask2] = (1/h**2) * (8/3 * R[mask2] - 3 * R[mask2]**2 + 6/5 * R[mask2]**3 - 
                                    1/6 * R[mask2]**4 - 1/(15 * R[mask2]**2))
        
        # Condition R ≥ 2 (Newtonian gravity)
        mask3 = R >= 2
        dphi_dr[mask3] = 1 / (r[mask3]**2 + 1e-12)
        
        return dphi_dr
        
    def equation_of_state(self, rho, e):
        return (self.gamma - 1.0) * rho * e
        
    def sound_speed(self, e):
        return np.sqrt((self.gamma - 1.0) * np.maximum(e, 1e-10))
        
    def artificial_viscosity_3d(self, v_i, v_j, x_i, x_j, rho_i, rho_j, c_i, c_j, h_ij):

        v_ij = v_i - v_j
        x_ij = x_i - x_j
        r_ij = np.linalg.norm(x_ij)
        
        if np.dot(v_ij, x_ij) >= 0:
            return 0.0
            
        phi_const = 0.1 * h_ij
        phi_ij = (h_ij * np.dot(v_ij, x_ij)) / (r_ij**2 + phi_const**2)
        
        rho_avg = 0.5 * (rho_i + rho_j)
        c_avg = 0.5 * (c_i + c_j)
        
        # viscosity
        Pi_ij = (-self.alpha * c_avg * phi_ij + self.beta * phi_ij**2) / rho_avg
        
        return Pi_ij
    
    def sph_equations_3d(self, t, y):

        self.call_count += 1
        
        if self.call_count % 100 == 0 and self.start_time is not None:
            elapsed = time.time() - self.start_time
            print(f"t={t:.6f}, calls={self.call_count:,}, elapsed={elapsed:.1f}s")
        
        x, v, rho, e = self._unpack_state(y)
        p = self.equation_of_state(rho, e)
        c = self.sound_speed(e)
        
        dx_dt = v.copy()
        dv_dt = np.zeros_like(v)
        de_dt = np.zeros_like(e)
        
        for i in tqdm(range(self.N), desc="SPH particles", leave=False):
            for j in range(self.N):
                if i == j:
                    continue
                
                x_ij = x[i] - x[j]
                r_ij = np.linalg.norm(x_ij)
                
                # Skip if beyond support (2h)
                if r_ij > 2.0 * self.h:
                    continue
                
                # Kernel gradient
                grad_W_ij = self.kernel_gradient_3d(x_ij, self.h)
                
                h_ij = self.h
                
                # viscosity
                Pi_ij = self.artificial_viscosity_3d(v[i], v[j], x[i], x[j], 
                                                   rho[i], rho[j], c[i], c[j], h_ij)
                v_ij = v[i] - v[j]
                
                # Momentum equation (hydro + viscosity)
                pressure_term = p[i]/(rho[i]**2) + p[j]/(rho[j]**2) + Pi_ij
                dv_dt[i] -= self.m[j] * pressure_term * grad_W_ij
                
                # Energy equation
                de_dt[i] += 0.5 * self.m[j] * pressure_term * np.dot(v_ij, grad_W_ij)
                
                # Gravity contribution
                if r_ij > 0:
                    dphi_dr = self.gravitational_potential_gradient(r_ij, self.h)
                    gravity_force = -self.G * self.m[j] * dphi_dr * x_ij / (r_ij + 1e-12)
                    dv_dt[i] += gravity_force
        
        max_accel = 1e6
        dv_dt = np.clip(dv_dt, -max_accel, max_accel)
        de_dt = np.clip(de_dt, -max_accel, max_accel)
        
        dy_dt = np.concatenate([dx_dt.flatten(), dv_dt.flatten(), np.zeros(self.N), de_dt])
        
        return dy_dt
    
    def solve(self, t_final=1.0, dt_max=1e-4, method='DOP853'):

        print(f"\nStarting 3D SPH integration:")
        print(f"  Final time: {t_final}")
        print(f"  Max time step: {dt_max}")
        print(f"  Method: {method}")
        
        self.start_time = time.time()
        self.call_count = 0
        
        t_span = (0, t_final)
        t_eval = np.linspace(0, t_final, 101)
        
        sol = solve_ivp(
            self.sph_equations_3d,
            t_span, 
            self.y0,
            method=method,
            t_eval=t_eval,
            max_step=dt_max,
            rtol=1e-6,
            atol=1e-8
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
    
    def plot_3d_solution(self, sol, save_frames=False):
        """Plot 3D planetary collision results"""
        if not sol.success:
            print("Cannot plot - integration failed")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        for i, t_idx in enumerate([0, len(sol.t)//2, -1]):
            state = sol.y[:, t_idx]
            x, v, rho, e = self._unpack_state(state)
            p = self.equation_of_state(rho, e)
            
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
            scatter = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=rho, cmap='viridis', s=10)
            ax.set_title(f't = {sol.t[t_idx]:.3f}s')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.colorbar(scatter, ax=ax, label='Density')
        
        for i, t_idx in enumerate([0, len(sol.t)//2, -1]):
            state = sol.y[:, t_idx]
            x, v, rho, e = self._unpack_state(state)
            
            ax = fig.add_subplot(2, 3, i+4)
            speed = np.linalg.norm(v, axis=1)
            ax.scatter(x[:, 0], x[:, 1], c=speed, cmap='plasma', s=10)
            ax.set_title(f'XY Projection, t = {sol.t[t_idx]:.3f}s')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(ax.collections[0], ax=ax, label='Velocity Magnitude')
        
        plt.tight_layout()
        plt.show()
    
    def create_animation(self, sol, filename='planetary_collision.mp4'):

        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            state = sol.y[:, frame]
            x, v, rho, e = self._unpack_state(state)
            
            scatter = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=rho, cmap='viridis', s=10)
            ax.set_title(f'Planetary Collision - t = {sol.t[frame]:.3f}s')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)
            
            return scatter,
        
        ani = FuncAnimation(fig, update, frames=len(sol.t), interval=50, blit=False)
        ani.save(filename, writer='ffmpeg', fps=20)
        plt.close()


def load_planet_data(filename):
    data = np.loadtxt(filename)
    
    particles = {
        'x': data[:, 0:3],      # positions
        'v': data[:, 3:6],      # velocities
        'm': data[:, 6],        # masses
        'rho': data[:, 7],      # densities
        'e': data[:, 8] / (1.4 - 1.0) / data[:, 7]  # internal energy from pressure
    }
    
    return particles


def create_two_planets(planet_data, separation=3.0, relative_velocity=1.0):

    # Copy first planet
    particles1 = planet_data.copy()
    
    # Create second planet (mirror position, opposite velocity)
    particles2 = {}
    for key in planet_data:
        if key == 'x':
            particles2[key] = planet_data[key].copy() + np.array([separation, 0, 0])
        elif key == 'v':
            particles2[key] = planet_data[key].copy() + np.array([-relative_velocity, 0, 0])
        else:
            particles2[key] = planet_data[key].copy()
    
    # Combine both planets
    combined = {}
    for key in planet_data:
        combined[key] = np.concatenate([particles1[key], particles2[key]])
    
    return combined


def add_planet_spin(particles, center, angular_velocity):

    x = particles['x']
    r = x - center
    spin_velocity = np.cross(angular_velocity, r)
    particles['v'] += spin_velocity
    return particles


# Example usage
if __name__ == "__main__":
    print("3D Planetary Collision SPH Simulation")
    
    # Load planet data
    planet_data = load_planet_data('planet300.dat')
    
    N = 200
    
    # Create a sphere of particles
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.random.uniform(0, np.pi, N)
    r = np.random.uniform(0, 1, N)
    
    x = np.array([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ]).T
    
    test_particles = {
        'x': x,
        'v': np.zeros((N, 3)),
        'm': np.full(N, 1.0/N),
        'rho': np.full(N, 1.0),
        'e': np.full(N, 2.5)
    }
    
    # Create two planets
    particles = create_two_planets(test_particles, separation=2.0, relative_velocity=0.5)
    
    # Add spin
    angular_velocity = np.array([0, 0, 0.5])  # spin around z-axis
    particles = add_planet_spin(particles, np.array([-1, 0, 0]), angular_velocity)
    particles = add_planet_spin(particles, np.array([1, 0, 0]), -angular_velocity)
    
    # solver
    solver = SPH3DSolver(particles, gamma=1.4, alpha=1.0, beta=1.0, eta=1.5)
    
    solution = solver.solve(t_final=0.6, dt_max=1e-3, method='RK45')
    
    # Plot results
    if solution.success:
        solver.plot_3d_solution(solution)
        solver.create_animation(solution, 'planetary_collision_test.gif')
    
    print("3D planetary collision simulation complete!")

