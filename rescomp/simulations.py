# -*- coding: utf-8 -*-
""" Simulate various chaotic system to generate artificial data """

import numpy as np
from . import utilities


def _roessler(x, a=0.5, b=2, c=4):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        a (float): 'a' parameter in the Roessler equations
        b (float): 'b' parameter in the Roessler equations
        c (float): 'c' parameter in the Roessler equations

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """

    np.array(x)
    if x.shape == (3,):
        return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])
    else:
        raise Exception('check shape of x, should have 3 components')


def _roessler_sprott(x, a=0.2, b=0.2, c=5.7):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    This version is identical to roessler(), but uses the parameters from
    Sprott as default.

    Args:
        x (np.ndarray): (x,y,z) coordinates
        a (float): 'a' parameter in the Roessler equations
        b (float): 'b' parameter in the Roessler equations
        c (float): 'c' parameter in the Roessler equations

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    np.array(x)
    if x.shape == (3,):
        return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])
    else:
        raise Exception('check shape of x, should have 3 components')


def _normal_lorenz(x, sigma=10, rho=28, beta=8/3):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): (x,y,z) coordinates
        sigma (float): 'sigma' parameter in the Lorenz 63 equations
        rho (float): 'rho' parameter in the Lorenz 63 equations
        beta (float): 'beta' parameter in the Lorenz 63 equations

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    np.array(x)
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]),
                         x[0] * (rho - x[2]) - x[1],
                         x[0] * x[1] - beta * x[2]])
    else:
        raise Exception('check shape of x, should have 3 components')


def _mod_lorenz(x, sigma=10, rho=28, beta=8/3):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    with dz/dt += x(t) to break symmetry

    Args:
        x (np.ndarray): (x,y,z) coordinates
        sigma (float): 'sigma' parameter in the Lorenz 63 equations
        rho (float): 'rho' parameter in the Lorenz 63 equations
        beta (float): 'beta' parameter in the Lorenz 63 equations

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    Examples:
        >>> _mod_lorenz(np.array([1.,2.,3.]))
        array([10., 23., -5.])

    """
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]),
                         x[0] * (rho - x[2]) - x[1],
                         x[0] * x[1] - beta * x[2] + x[0]])
    else:
        raise Exception('check shape of x, should have 3 components')


def _mod_lorenz_wrong(x, sigma=10, rho=28, beta=8 / 3):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    with dz/dt += x(t) to break symmetry

    Args:
        x (np.ndarray): (x,y,z) coordinates
        sigma (float): 'sigma' parameter in the Lorenz 63 equations
        rho (float): 'rho' parameter in the Lorenz 63 equations
        beta (float): 'beta' parameter in the Lorenz 63 equations

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    np.array(x)
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]),
                         x[0] * (rho - x[2]) - x[1],
                         x[0] * x[1] - beta * x[2]] + x[0])
    else:
        raise Exception('check shape of x, should have 3 components')


def _chua(x):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): (x,y,z) coordinates

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    alpha=9.
    beta=100./7.
    a=8./7.
    b=5./7.

    np.array(x)
    if x.shape == (3,):
        return np.array([alpha*(x[1]-x[0]+b*x[0]+0.5*(a-b)*(np.abs(x[0]+1)
                        -np.abs(x[0]-1))),x[0]-x[1]+x[2], -beta*x[1]])
    else:
        raise Exception('check shape of x, should have 3 components')


# def g_chua(x):
#     """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4
#
#     for Chua ODEs
#
#     Args:
#         x (np.ndarray): (x,y,z) coordinates
#
#     Returns:
#         (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x
#
#     """
#    m0=-0.5
#    m1=-0.8
#    Bp=1.
#    return m0*x+0.5*(m1-m0)*(np.abs(x+Bp)-np.abs(x-Bp))


def _ueda(x):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): (x,y,z) coordinates

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    delta = 0.05
    beta = 0.
    alpha = 1.
    gamma=7.5
    omega=1.
    np.array(x)
    if x.shape == (3,):
        return np.array([x[1],
                -delta*x[1] - beta*x[0] - alpha*x[0]**3 + gamma*np.cos(x[2]),
                omega])
    else:
        raise Exception('check shape of x, should have 3 components')


def _complex_butterfly(x):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): (x,y,z) coordinates

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    a=0.55
    np.array(x)
    if x.shape == (3,):
        return np.array([a*(x[1]-x[0]), -x[2]*np.sign(x[0]), np.abs(x[0]) - 1])
    else:
        raise Exception('check shape of x, should have 3 components')


def _chen(x):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): (x,y,z) coordinates

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    a = 35.
    b = 3.
    c = 28.
    np.array(x)
    if x.shape == (3,):
        return np.array([a * (x[1] - x[0]), (c-a)*x[0] -x[0]*x[2]+c*x[1], x[0] * x[1] - b * x[2]])
    else:
        raise Exception('check shape of x, should have 3 components')


def _rucklidge(x):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): (x,y,z) coordinates

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    kappa = 2.
    lam = 6.7
    
    np.array(x)
    if x.shape == (3,):
        return np.array([-kappa*x[0] + lam*x[1] - x[1]*x[2],
                         x[0], -x[2] + x[1]**2])
    else:
        raise Exception('check shape of x, should have 3 components')


def _rabinovich(x):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): (x,y,z) coordinates

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    alpha = 1.1
    gamma = 0.87
    
    np.array(x)
    if x.shape == (3,):
        return np.array([x[1]*(x[2]-1+x[0]**2)+gamma*x[0],
                         x[0]*(3*x[2]+1-x[0]**2)+gamma*x[1],
                         -2*x[2]*(alpha + x[1]*x[0])])
    else:
        raise Exception('check shape of x, should have 3 components')


def _thomas(x):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): (x,y,z) coordinates

    Returns:
        (np.ndarray): (dx/dt, dy/dt, dz/dt) corresponding to input x

    """
    b=0.18
    np.array(x)
    if x.shape == (3,):
        return np.array([-b*x[0]+np.sin(x[1]), -b*x[1]+np.sin(x[2]), -b*x[2]+np.sin(x[0])])
    else:
        raise Exception('check shape of x, should have 3 components')


#TODO: Rewrite using numpy vectorization to make faster
def _lorenz_96(x, force=8):
    """ Calculates (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4

    Args:
        x (np.ndarray): d-dim coordinates
        force (float): force parameter in the Lorenz96 equations

    Returns:
        (np.ndarray): d-dim time derivative at x

    """
    # compute state derivatives
    system_dimension = x.shape[0]
    derivative = np.zeros(system_dimension)

    # Periodic Boundary Conditions for the 3 edge cases i=1,2,system_dimension
    derivative[0] = (x[1] - x[system_dimension - 2]) * x[system_dimension - 1] - x[0]
    derivative[1] = (x[2] - x[system_dimension - 1]) * x[0] - x[1]
    derivative[system_dimension - 1] = (x[0] - x[system_dimension - 3]) * x[system_dimension - 2] - x[system_dimension - 1]

    # then the general case
    for i in range(2, system_dimension - 1):
        derivative[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]

    # add the forcing term
    derivative = derivative + force

    # return the state derivatives
    return derivative


def _runge_kutta(f, dt, y=np.array([2.2, -3.5, 4.3])):
    """ Simulate one step for ODEs of the form dy/dt = f(t,y), returns y(t + dt)

    Args:
        f (fct): function used to calculate the time derivate at point y
        dt (float): time step size
        y (np.ndarray): d-dim position at time t

    Returns:
        (np.ndarray): d-dim position at time t+dt

    """

    # print(y)
    k1 = dt * f(y)
    k2 = dt * f(y + k1 / 2)
    k3 = dt * f(y + k2 / 2)
    k4 = dt * f(y + k3)
    return y + 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# If you add/change a flag here, please also change it in the
# simulate_trajectory docstring
_sys_flag_synonyms = utilities._SynonymDict()
_sys_flag_synonyms.add_synonyms(0, "mod_lorenz")
_sys_flag_synonyms.add_synonyms(1, "mod_lorenz_wrong")
_sys_flag_synonyms.add_synonyms(2, ["lorenz_63", "normal_lorenz", "lorenz"])
_sys_flag_synonyms.add_synonyms(3, "roessler")
_sys_flag_synonyms.add_synonyms(4, "lorenz_96")
_sys_flag_synonyms.add_synonyms(5, "ueda")
_sys_flag_synonyms.add_synonyms(6, "chua")
_sys_flag_synonyms.add_synonyms(7, "complex_butterfly")
_sys_flag_synonyms.add_synonyms(8, "chen")
_sys_flag_synonyms.add_synonyms(9, "rucklidge")
_sys_flag_synonyms.add_synonyms(10, "rabinovich")
_sys_flag_synonyms.add_synonyms(11, "thomas")
_sys_flag_synonyms.add_synonyms(12, "roessler_sprott")
_sys_flag_synonyms.add_synonyms(13, "kuramoto_sivashinsky")
_sys_flag_synonyms.add_synonyms(14, "kuramoto_sivashinsky_old")
_sys_flag_synonyms.add_synonyms(15, "kuramoto_sivashinsky_custom")

def simulate_trajectory(sys_flag='mod_lorenz', dt=2e-2, time_steps=int(2e4),
                        starting_point=None, **kwargs):
    """ Simulate a trajectory in an artificial chaotic system

    Args:
        sys_flag (str): The system to be simulated. Possible flags, their
            synonyms and corresponding possible kwargs are:

            - "lorenz_63", "normal_lorenz", "lorenz": The normal, unmodified
              Lorenz-63 system. Possible kwargs:
                - sigma (float): 'sigma' parameter in the Lorenz 63 equations
                - rho (float): 'rho' parameter in the Lorenz 63 equations
                - beta (float): 'beta' parameter in the Lorenz 63 equations
            - "roessler": The normal, unmodified Roessler system. Possible
              kwargs:
                - a (float): 'a' parameter in the Roessler equations
                - b (float): 'b' parameter in the Roessler equations
                - c (float): 'c' parameter in the Roessler equations
            - "mod_lorenz": Modified Lorenz system. Possible kwargs:
                - sigma (float): 'sigma' parameter in the Lorenz 63 equations
                - rho (float): 'rho' parameter in the Lorenz 63 equations
                - beta (float): 'beta' parameter in the Lorenz 63 equations
            - "mod_lorenz_wrong": Incorrectly modified Lorenz system, kept for
              backward compatibility. Possible kwargs:
                - sigma (float): 'sigma' parameter in the Lorenz 63 equations
                - rho (float): 'rho' parameter in the Lorenz 63 equations
                - beta (float): 'beta' parameter in the Lorenz 63 equations
            - "lorenz_96": The d-dimensional Lorenz-96 System. Possible kwargs:
                - force (float): force parameter in the Lorenz96 equations
            - "roessler_sprott". Identical to "roessler", uses the parameters
              from Sprott as default. Possible kwargs:
                - a (float): 'a' parameter in the Roessler equations
                - b (float): 'b' parameter in the Roessler equations
                - c (float): 'c' parameter in the Roessler equations
            - "kuramoto_sivashinsky". The d-dimensional Lorenz-96 System. Note
              that, due to the way the KS system is simulated, the
              "starting_point" parameter, does not have any effect! The system's
              dimension is instead set by the following possible kwargs:
                - dimensions (int): nr. of dimensions, d, of the system grid.
                  The output will have shape (T, d).
                - system_size (int): 'physical' size of the system
            - "ueda". Possible kwargs:
                - None
            - "chua". Possible kwargs:
                - None
            - "complex_butterfly". Possible kwargs:
                - None
            - "chen". Possible kwargs:
                - None
            - "rucklidge". Possible kwargs:
                - None
            - "rabinovich". Possible kwargs:
                - None
            - "thomas". Possible kwargs:
                - None

        dt (float): Size of time steps
        time_steps (int): Number of time steps to simulate
        starting_point (np.ndarray): Starting point of the trajectory. Typically
            3-dimensional unless otherwise specified.
        **kwargs (): Further Arguments passed to the simulating function,
            usually not needed. See above for a list of possible arguments

    Returns:
        trajectory (np.ndarray) the full trajectory, ready to be used for RC

    """

    sys_flag_syn = _sys_flag_synonyms.get_flag(sys_flag)

    #TODO: this is bad, as it's removed from the actual function call below.
    if starting_point is None and sys_flag_syn <= 12:
        starting_point = np.array([1, 2, 3])

    if sys_flag_syn == 0:
        f = lambda x: _mod_lorenz(x, **kwargs)
    elif sys_flag_syn == 1:
        # TODO: should be a warning in the logger.
        print('YOU ARE USING AN UNUSUAL KIND OF LORENZ EQUATION! USE WITH CARE')
        f = lambda x: _mod_lorenz_wrong(x, **kwargs)
    elif sys_flag_syn == 2:
        f = lambda x: _normal_lorenz(x, **kwargs)
    elif sys_flag_syn == 3:
        f = lambda x: _roessler(x, **kwargs)
    elif sys_flag_syn == 4:
        # Starting point is ignored here atm
        if starting_point is not None:
            #TODO: should be a warning in the logger.
            print("Starting point is ignored for the Lorenz96 equation")
        f = lambda x: _lorenz_96(x, **kwargs)
    elif sys_flag_syn == 5:
        f = lambda x: _ueda(x)
    elif sys_flag_syn == 6:
        f = lambda x: _chua(x)
    elif sys_flag_syn == 7:
        f = lambda x: _complex_butterfly(x)
    elif sys_flag_syn == 8:
        f = lambda x: _chen(x)
    elif sys_flag_syn == 9:
        f = lambda x: _rucklidge(x)
    elif sys_flag_syn == 10:
        f = lambda x: _rabinovich(x)
    elif sys_flag_syn == 11:
        f = lambda x: _thomas(x)
    elif sys_flag_syn == 12:
        f = lambda x: _roessler_sprott(x, **kwargs)
    elif sys_flag_syn == 13:
        return _kuramoto_sivashinsky(dt=dt, time_steps=time_steps - 1, starting_point=starting_point, **kwargs)
    elif sys_flag_syn == 14:
        if not starting_point is None:
            # TODO: should be a warning in the logger.
            print("WARNING starting point is ignored for this simulation fct!")
        return _kuramoto_sivashinsky_old(dt=dt, time_steps=time_steps - 1, **kwargs)
    elif sys_flag_syn == 15:
        return _kuramoto_sivashinsky_custom(dt=dt, time_steps=time_steps - 1, starting_point=starting_point, **kwargs)
    else:
        raise Exception('sys_flag not recoginized')

    traj_size = ((time_steps, starting_point.shape[0]))
    traj = np.zeros(traj_size)
    y = starting_point

    for t in range(traj_size[0]):
        traj[t] = y
        y = _runge_kutta(f, dt, y=y)
    return traj


def _kuramoto_sivashinsky_old(dimensions, system_size, dt, time_steps):
    """ This function INCORRECTLY simulates the Kuramoto–Sivashinsky PDE

    It is kept here only for historical reasons.
    DO NOT USE UNLESS YOU WANT INCORRECT RESULTS

    Even though it doesn't use the RK4 algorithm, it is bundled with the other
    simulation functions in simulate_trajectory() for consistency.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    Args:
        dimensions (int): nr. of dimensions of the system grid
        system_size (int): physical size of the system
        dt (float): time step size
        time_steps (int): nr. of time steps to simulate

    Returns:
        (np.ndarray): simulated trajectory of shape (time_steps, dimensions)

    """
    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  #

    # Define initial conditions and Fourier Transform them
    x = np.transpose(np.conj(np.arange(1, n + 1))) / n
    u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    v = np.fft.fft(u)

    h = dt  # time step
    nmax = time_steps # No. of time steps to simulate
    # Wave numbers
    k = np.transpose(
        np.conj(np.concatenate((np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0))))) * 2 * np.pi / size

    # Just copied from the paper, it works
    L = k ** 2 - k ** 4
    E = np.exp(h * L)
    E_2 = np.exp(h * L / 2)
    M = 16
    # M = (size * np.pi) //2
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

    uu = [np.array(u)]  # List of Real space solutions, later converted to a np.array

    g = -0.5j * k

    # See paper for details
    for n in range(1, nmax + 1):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E_2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E_2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E_2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        u = np.real(np.fft.ifft(v))
        uu.append(np.array(u))

    uu = np.array(uu)
    # print("PDE simulation finished")

    return uu


def _kuramoto_sivashinsky(dimensions, system_size, dt, time_steps, starting_point, **kwargs):
    """ This function simulates the Kuramoto–Sivashinsky PDE

    Even though it doesn't use the RK4 algorithm, it is bundled with the other
    simulation functions in simulate_trajectory() for consistency.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    Args:
        dimensions (int): nr. of dimensions of the system grid
        system_size (int): physical size of the system
        dt (float): time step size
        time_steps (int): nr. of time steps to simulate
        starting_point (np.ndarray): starting point for the simulation of shape
            (dimensions, )

    Returns:
        (np.ndarray): simulated trajectory of shape (time_steps, dimensions)

    """
    # Rename variables to the names used in the paper
    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # system size
    h = dt  # time step
    nmax = time_steps  # No. of time steps to simulate

    # Define initial conditions and Fourier Transform them
    if starting_point is None:
        # Use the starting point from the Kassam_2005 paper
        x = size * np.transpose(np.conj(np.arange(1, n + 1))) / n
        u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    else:
        u = starting_point
    v = np.fft.fft(u)

    # Wave numbers
    k = np.transpose(np.conj(np.concatenate((
                np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0))
                ))) * 2 * np.pi / size

    L = k ** 2 - k ** 4
    E = np.exp(h * L)
    E_2 = np.exp(h * L / 2)
    M = 64  # changed because the previous optimizations were negligable and could result in errors for small size systems
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(
        np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3,
                axis=1))
    f2 = h * np.real(
        np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = h * np.real(
        np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3,
                axis=1))

    # List of Real space solutions, later converted to a np.array
    uu = [np.array(u)]

    g = -0.5j * k

    # See paper for details
    for n in range(1, nmax + 1):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E_2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E_2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E_2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

        u = np.real(np.fft.ifft(v))
        uu.append(np.array(u))

    uu = np.array(uu)
    # print("PDE simulation finished")

    return uu


def _kuramoto_sivashinsky_custom(dimensions, system_size, dt, time_steps, starting_point,
                                 precision=None, fft_type=None, **kwargs):
    """ This function simulates the Kuramoto–Sivashinsky PDE with custom precision and fft backend"""

    if precision is None:
        change_precision = False
    elif precision == 128:
        change_precision = True
        f_dtype = 'float128'
        c_dtype = 'complex256'
    elif precision == 64:
        change_precision = True
        f_dtype = 'float64'
        c_dtype = 'complex128'
    elif precision == 32:
        change_precision = True
        f_dtype = 'float32'
        c_dtype = 'complex64'
    elif precision == 16:
        change_precision = True
        f_dtype = 'float16'
        c_dtype = 'complex32'
    else:
        raise Exception("specified precision not recognized")

    if fft_type is None or fft_type == 'numpy':
        custom_fft = np.fft.fft
        custom_ifft = np.fft.ifft
    elif fft_type == 'scipy':
        import scipy
        try:
            import scipy.fft
            custom_fft = scipy.fft.fft
            custom_ifft = scipy.fft.ifft
        except ModuleNotFoundError:
            # Depricated, but needed for older versions of scipy
            import scipy.fftpack
            custom_fft = scipy.fftpack.fft
            custom_ifft = scipy.fftpack.ifft
    elif fft_type == 'pyfftw_np':
        import pyfftw
        custom_fft = pyfftw.interfaces.numpy_fft.fft
        custom_ifft = pyfftw.interfaces.numpy_fft.ifft
    elif fft_type == 'pyfftw_sc':
        import pyfftw
        custom_fft = pyfftw.interfaces.scipy_fft.fft
        custom_ifft = pyfftw.interfaces.scipy_fft.ifft
    elif fft_type == 'pyfftw_fftw':
        import pyfftw
        a = pyfftw.empty_aligned(dimensions, dtype='complex128')
        b = pyfftw.empty_aligned(dimensions, dtype='complex128')
        c = pyfftw.empty_aligned(dimensions, dtype='complex128')
        fft_object = pyfftw.FFTW(a, b)
        ifft_object = pyfftw.FFTW(b, c, direction='FFTW_BACKWARD')
        custom_fft = fft_object
        custom_ifft = ifft_object
    else:
        raise Exception('fft_type not recognized')

    # Rename variables to the names used in the paper
    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # system size
    h = dt  # time step
    nmax = time_steps  # No. of time steps to simulate

    # Define initial conditions and Fourier Transform them
    if starting_point is None:
        # Use the starting point from the Kassam_2005 paper
        x = size * np.transpose(np.conj(np.arange(1, n + 1))) / n
        u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    else:
        u = starting_point
    if change_precision: u = u.astype(np.single)
    v = custom_fft(u)

    # Wave numbers
    k = np.transpose(np.conj(np.concatenate((
                np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0))
        ))) * 2 * np.pi / size
    if change_precision: k = k.astype(f_dtype)

    L = k ** 2 - k ** 4
    E = np.exp(h * L)
    if change_precision: E = E.astype(f_dtype)
    E_2 = np.exp(h * L / 2)
    if change_precision: E_2 = E_2.astype(f_dtype)
    M = 64
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    if change_precision: r = r.astype(c_dtype)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    if change_precision: LR = LR.astype(c_dtype)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    if change_precision: Q = Q.astype(c_dtype)
    f1 = h * np.real(
        np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3,
                axis=1))
    if change_precision: f1 = f1.astype(c_dtype)
    f2 = h * np.real(
        np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    if change_precision: f2 = f2.astype(c_dtype)
    f3 = h * np.real(
        np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3,
                axis=1))
    if change_precision: f3 = f3.astype(c_dtype)

    # List of Real space solutions, later converted to a np.array
    uu = [np.array(u)]

    g = -0.5j * k
    if change_precision: g = g.astype(c_dtype)

    # See paper for details
    for n in range(1, nmax + 1):
        Nv = g * custom_fft(np.real(custom_ifft(v)) ** 2)
        if change_precision: Nv = Nv.astype(c_dtype)
        a = E_2 * v + Q * Nv
        if change_precision: a = a.astype(c_dtype)
        Na = g * custom_fft(np.real(custom_ifft(a)) ** 2)
        if change_precision: Na = Na.astype(c_dtype)
        b = E_2 * v + Q * Na
        if change_precision: b = b.astype(c_dtype)
        Nb = g * custom_fft(np.real(custom_ifft(b)) ** 2)
        if change_precision: Nb = Nb.astype(c_dtype)
        c = E_2 * a + Q * (2 * Nb - Nv)
        if change_precision: c = c.astype(c_dtype)
        Nc = g * custom_fft(np.real(custom_ifft(c)) ** 2)
        if change_precision: Nc = Nc.astype(c_dtype)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        # v = E * v
        # v += Nv * f1
        # v += 2 * (Na + Nb) * f2
        # v += Nc * f3
        if change_precision: v = v.astype(c_dtype)
        u = np.real(custom_ifft(v))
        uu.append(np.array(u))

    uu = np.array(uu)
    # print("PDE simulation finished")

    return uu
