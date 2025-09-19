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
        # self.X_0 = X_0
        # self.delta_t = delta_t
        # ...

        pass

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