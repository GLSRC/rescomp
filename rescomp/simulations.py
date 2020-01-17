# -*- coding: utf-8 -*-
""" Calculating the lorenz63-attractor and other chaotic systems using 4th order runge kutta method

@author: aumeier, baur, mabrouk
"""

import numpy as np


def roessler(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    a = 0.5
    b = 2.
    c = 4.
    np.array(x)
    if x.shape == (3,):
        return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])
    else:
        raise Exception('check shape of x, should have 3 components')


def normal_lorenz(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    sigma = 10.
    rho = 28.
    b = 8 / 3
    np.array(x)
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - b * x[2]])
    else:
        raise Exception('check shape of x, should have 3 components')


def mod_lorenz(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    with dz/dt += x(t) to break symmetry
    
    >>> mod_lorenz(np.array([1.,2.,3.]))
    array([10., 23., -5.])
    '''
    sigma = 10.
    rho = 28.
    b = 8 / 3
    # np.array(x)
    # print('cheese')
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - b * x[2] + x[0]])
    else:
        raise Exception('check shape of x, should have 3 components')


def mod_lorenz_wrong(x):
    '''
    
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    with dz/dt += x(t) to break symmetry
    '''
    sigma = 10.
    rho = 28.
    b = 8 / 3
    np.array(x)
    if x.shape == (3,):
        return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - b * x[2]] + x[0])
    else:
        raise Exception('check shape of x, should have 3 components')


# def lorenz_96(x, dim=11, force=8):
#     # compute state derivatives
#     der = np.zeros(dim)
#
#     # Periodic Boundary Conditions for the 3 edge cases i=1,2,system_dimension
#     der[0] = (x[1] - x[dim - 2]) * x[dim - 1] - x[0]
#     der[1] = (x[2] - x[dim - 1]) * x[0] - x[1]
#     der[dim - 1] = (x[0] - x[dim - 3]) * x[dim - 2] - x[dim - 1]
#
#     # then the general case
#     for i in range(2, dim - 1):
#         der[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
#
#     # add the forcing term
#     der = der + force
#
#     # return the state derivatives
#     return der

#TODO: Rewrite using numpy vectorization to make faster
def lorenz_96(x, system_dimension=11, force=8):
    # compute state derivatives
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

def runge_kutta(f, dt, y=np.array([2.2, -3.5, 4.3])):
    '''
    the function approximates differential equations of the form dy/dt = f(t,y)
    returns y(t + dt)
    '''
    # print(y)
    k1 = dt * f(y)
    k2 = dt * f(y + k1 / 2)
    k3 = dt * f(y + k2 / 2)
    k4 = dt * f(y + k3)
    return y + 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def record_trajectory(sys_flag='mod_lorenz', dt=2e-2, timesteps=int(2e4),
                      print_switch=False, starting_point=None, **kwargs):
    if print_switch:
        print(sys_flag)

    if sys_flag == 'mod_lorenz':
        f = mod_lorenz
    elif sys_flag == 'mod_lorenz_wrong':
        print('YOU ARE USING AN UNUSUAL KIND OF LORENZ EQUATION! USE WITH CARE')
        f = mod_lorenz_wrong
    elif sys_flag == 'normal_lorenz':
        f = normal_lorenz
    elif sys_flag == 'roessler':
        f = roessler
    elif sys_flag == 'lorenz_96':
        f = lambda x: lorenz_96(x, **kwargs)
    else:
        raise Exception('sys_flag not recoginized')

    traj_size = ((timesteps, starting_point.shape[0]))
    traj = np.zeros(traj_size)
    if print_switch:
        print('record_trajector received and used: starting_point: ', starting_point)
    y = starting_point

    for t in range(traj_size[0]):
        traj[t] = y
        y = runge_kutta(f, dt, y=y)
    return traj

# not used currently:
# def save_trajectory(timesteps, dt): #starting_point=np.array([-13.5,10.8,-17.9])
#     np.savetxt('lorenz_attractor_'+str(timesteps)+'_a_'+str(dt)+'.dat', record_trajectory(timesteps=timesteps, dt=dt))


# TODO: Refactor to make more legible, then add a docstring, remove/add print statements etc.
def ks_pde_simulation(dimensions, system_size, t_max, time_step):
    # This function simulates the Kuramoto–Sivashinsky PDE
    # reference for the numerical integration : "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    # https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
    # Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    # print("Start PDE simulation")

    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # grid points for the PDE simulation #TODO: I think

    # Define initial conditions and Fourier Transform them
    x = np.transpose(np.conj(np.arange(1, n + 1))) / n
    u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    v = np.fft.fft(u)

    h = time_step  # time step

    # Wave numbers
    k = np.transpose(
        np.conj(np.concatenate((np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0))))) * 2 * np.pi / size

    # Just copied from the paper, it works
    L = k ** 2 - k ** 4
    E = np.exp(h * L)
    E_2 = np.exp(h * L / 2)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

    uu = [np.array(u)]  # List of Real space solutions, later converted to a np.array
    tmax = t_max  # Length of time to simulate
    nmax = round(tmax / h)  # No. of time steps to simulate

    g = -0.5j * k  # TODO: Meaning?

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
