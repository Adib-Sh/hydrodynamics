import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

class SPHSolver:
    def __init__(self, X_0, gamma=1.4, alpha=1.0, beta=1.0, eta=1.2, dt=0.005, n_steps=40):

        self.pos = X_0[:, 0]
        self.v = X_0[:, 3]
        self.m   = X_0[:, 6]
        self.rho = X_0[:, 7]
        self.e   = X_0[:, 9]
    
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.dt = dt
        self.n_steps = n_steps
        self.N = len(self.pos)
        self.h = 0.0045 #eta * np.mean(self.m / self.rho)
        self.X_0_flat = np.concatenate([self.pos, self.v, self.rho, self.e])
    
   
    # kernel function
    def c_kernel(self, r, h):
    
        alpha = 1.0 / h
        R = abs(r) / h
        W  = np.zeros_like(R)
        
        m1 = (R >= 0) & (R < 1)
        m2 = (R >= 1) & (R < 2)
        # calculating using the R conditions

        W[m1] = alpha * (2 / 3 - R[m1]**2 + 0.5 * R[m1]**3)
        W[m2] = alpha * (1 / 6) * (2 - R[m2])**3
  
        return W


    # kernel derivative function
    def c_kernel_dr(self, r, h):
        
        alpha = 1.0 / h**2
        R = np.abs(r) / h
        signR = np.sign(r)
        dW_dr = np.zeros_like(R)
        
        m1 = (R >= 0) & (R < 1)
        m2 = (R >= 1) & (R < 2)
    
        dW_dr[m1] = alpha * (-2*R[m1] + 1.5*R[m1]**2) * signR[m1]
        dW_dr[m2] = -alpha * 0.5*(2-R[m2])**2 * signR[m2]
    
        return dW_dr
    
   

    def f(self, t, y):
        N = self.N
        pos = y[0*N:1*N]
        v   = y[1*N:2*N] 
        rho = y[2*N:3*N]
        e   = y[3*N:4*N]

        p = (self.gamma - 1.0) * rho * e
    
        c = np.sqrt((self.gamma - 1.0) * e)
    

        xi = pos[:, None]    # (N,1)
        xj = pos[None, :]    # (1,N)
        xij = xi - xj        # (N,N)
        r = np.abs(xij)      # (N,N)
    
        # kernel and gradient
        W = self.c_kernel(r, self.h)
        dW_dx = self.c_kernel_dr(xij, self.h)
  
        # broadcasting arrays
        m_j = self.m[None, :]        # (1,N)
        rho_i = rho[:, None]     # (N,1)
        rho_j = rho[None, :]     # (1,N)
        p_i = p[:, None]
        p_j = p[None, :]
        v_i = v[:, None]
        v_j = v[None, :]
        vij = v_i - v_j              # (N,N)
        c_avg = 0.5 * (c[:, None] + c[None, :])
        rho_avg = 0.5 * (rho_i + rho_j)
    
        # viscosity
        eps = 0.1 * self.h
        vdotr = vij * xij
        phi_ij = np.zeros_like(vdotr)
        phi_ij = (self.h * vdotr) / (abs(r)**2 + eps**2)
        Pi = np.zeros_like(phi_ij)
        Pi = (-self.alpha * c_avg * phi_ij+ self.beta * phi_ij**2) / rho_avg
    
        # derivatives
        drho_dt = np.sum(m_j * vij * dW_dx, axis=1)
    
        term = (p_i / (rho_i**2)) + (p_j / (rho_j**2)) + Pi
        dv_dt = -np.sum(m_j * term * dW_dx, axis=1)
    
        de_dt = 0.5 * np.sum(m_j * term * vij * dW_dx, axis=1)
    
        dpos_dt = v
    
        # flatten everything
        dy = np.zeros(4 * N)
        dy[0*N:1*N] = dpos_dt
        dy[1*N:2*N] = dv_dt
        dy[2*N:3*N] = drho_dt
        dy[3*N:4*N] = de_dt
    
        return dy
        
        

    def solve(self, method='RK45'):
        t_span = (0, self.n_steps * self.dt)
        t_eval = np.linspace(0, self.n_steps * self.dt, self.n_steps)
        self.result = solve_ivp(self.f, t_span, self.X_0_flat,
                                method=method,
                                t_eval=t_eval,
                                max_step=self.dt,
                                rtol = 1e-4,
                                atol = 1e-6)
        return self.result 




if __name__ == "__main__":

    m = 0.001875
    x_min = -0.6
    x_max = 0.6

    N_hi = 320
    N_lo = 80

    p_hi = 1
    p_lo = 0.1795

    rho_hi = 1
    rho_lo = 0.25

    v_hi = 0.0
    v_lo = 0.0

    e_hi = 2.5
    e_lo = 1.795

    N = N_hi + N_lo

    x_hi = np.linspace(-0.6, 0, num=N_hi, endpoint=False)
    x_lo = np.linspace(0, 0.6, num=N_lo, endpoint=False)
    
    # Initial conditions
    x_0 = np.concatenate([x_hi, x_lo])
    p_0 = np.concatenate([np.full(N_hi, p_hi), np.full(N_lo, p_lo)])
    rho_0 = np.concatenate([np.full(N_hi, rho_hi), np.full(N_lo, rho_lo)])
    v_0 = np.concatenate([np.full(N_hi, v_hi), np.full(N_lo, v_lo)])
    e_0 = np.concatenate([np.full(N_hi, e_hi), np.full(N_lo, e_lo)])

    # initial vector X_0
    X_0 = np.zeros((N, 10))
    X_0[:, 0] = x_0            # position
    X_0[:, 3] = v_0            # velocity  
    X_0[:, 6] = np.full(N, m)  # mass
    X_0[:, 7] = rho_0          # density
    X_0[:, 8] = p_0            # pressure
    X_0[:, 9] = e_0            # internal energy

    # Create and run solver
    solver = SPHSolver(X_0, gamma=1.4, alpha=1.0, beta=1.0, eta=1.2, dt=0.005, n_steps=40)
    result = solver.solve(method='RK45')
