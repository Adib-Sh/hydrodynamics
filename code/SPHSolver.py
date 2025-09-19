class SPHSolver:
    """
    This is a solver for solving Smoothed Particle Hydrodynamics (SPH) problems.
    It uses Runge-Kutta (RK45) methods to solve modified Navier-Stokes equations.
    """
    def __init__(self, X_0):
        """
        When we create the SPHSolver instance, we assign all the user-set parameters and the
        initial condition here.
        """
        self.pos = X_0[:, :3]
        self.vel = X_0[:, 3:6]
        self.m   = X_0[:, 6]
        self.rho = X_0[:, 7]
        self.p   = X_0[:, 8]
        self.v   = X_0[:, 9]
        self.e   = X_0[:, 10]
        
        # self.delta_t = delta_t
        # ...

        pass
    
    # kernel function
    def c_kernel(r, h):
    
        alpha = 1.0 / h
        R = abs(r) / h
        
        # calculating using the R conditions
        if R >= 0 and R < 1:
            W = alpha * (2 / 3 - R**2 + 0.5 * R**3)
        elif R >= 1 and R < 2:
            W = alpha * (1 / 6) * (2.0 - R)**3
        else:
            W = 0.0
        
        return W

    # kernel derivative function
    def c_kernel_dr(r, h):
        
        alpha = 1.0 / h
        R = abs(r) / h
        
        
        # calculating using the R conditions
        if R >= 0 and R < 1:
            dW_dr = alpha * (-2 + 1.5 * R) * R / h
        elif R >= 1 and R < 2:
            dW_dr = -alpha * 0.5 * (2 - R)**2 * R / r
        else:
            dW_dr = 0.0
        
        return dW_dr
    
    # phi function
    def phi (h_avg, v_avg, xij):
        
        phi_const = 0.1 * h_avg
        
        
        phi_ij = h_avg * v_avg * xij / (abs(xij)**2 * phi_const)
        
        return phi_ij
    

    def f():
        """
        The modified Navier-Stokes equations that we want to integrate at each time step
        """

        pass

    def solve(self, method):
        """
        A function that takes our assigned parameters, and the initial condition, and then computes the result.

        Arguments:
            method  str     A string compatible with solve_ivp() which will determine the integration method used.
        """

        # ...
        # self.result = output from solve_ivp
        pass