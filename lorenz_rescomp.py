# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:35:55 2019

@author: aumeier
"""

'''
calculating the lorenz-attractor using 4th order runge kutta methond
'''
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
        return  np.array([-x[1]-x[2], x[0]+a*x[1], b+x[2]*(x[0]-c)])
    else:
        return 'check shape of x, should have 3 components'

def normal_lorenz(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    '''
    sigma = 10.
    rho = 28.
    b = 8/3
    np.array(x)
    if x.shape == (3,):
        return  np.array([sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1] - b*x[2]])
    else:
        return 'check shape of x, should have 3 components'

def mod_lorenz(x):
    '''
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    with dz/dt += x(t) to break symmetry
    '''
    sigma = 10.
    rho = 28.
    b = 8/3
    np.array(x)
    if x.shape == (3,):
        return  np.array([sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1] - b*x[2] + x[0]])
    else:
        return 'check shape of x, should have 3 components'

def mod_lorenz_wrong(x):
    '''
    
    returns (dx/dt, dy/dt, dz/dt) for given (x,y,z)
    with dz/dt += x(t) to break symmetry
    '''
    sigma = 10.
    rho = 28.
    b = 8/3
    np.array(x)
    if x.shape == (3,):
        return  np.array([sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1] - b*x[2]] + x[0])
    else:
        return 'check shape of x, should have 3 components'
        
def runge_kutta(f, dt, y=np.array([2.2,-3.5,4.3])):
    '''
    the function approximates differential equations of the form dy/dt = f(t,y)
    returns y(t + dt)
    '''
    #print(y)
    k1 = dt*f(y)
    k2 = dt*f(y+k1/2)
    k3 = dt*f(y+k2/2)
    k4 = dt*f(y + k3)
    return y + 1./6*(k1+2*k2+2*k3+k4)
    
def record_trajectory(sys_flag='mod_lorenz', dt=1., timesteps=int(10e4),
                      print_switch=False, starting_point=None):
    if print_switch:
        print(sys_flag)
    if sys_flag == 'mod_lorenz':
        f = mod_lorenz
    if sys_flag == 'mod_lorenz_wrong':
        print('YOU ARE USING A NON-USUAL KIND OF LORENZ EQUATION! USE WITH CARE')        
        f = mod_lorenz_wrong
    if sys_flag == 'normal_lorenz':
        f = normal_lorenz
    if sys_flag == 'roessler':
        f = roessler
        
    traj_size = ((timesteps, starting_point.shape[0]))
    traj = np.zeros(traj_size)
    if print_switch:
        print('record_trajector received and used: starting_point: ', starting_point)
    y = starting_point
    
    for t in range(traj_size[0]):
        traj[t] = y
        y = runge_kutta(f, dt, y=y)
    return traj
    
#not used currently:
def save_trajectory(timesteps, dt): #starting_point=np.array([-13.5,10.8,-17.9])
    np.savetxt('lorenz_attractor_'+str(timesteps)+'_a_'+str(dt)+'.dat', record_trajectory(timesteps=timesteps, dt=dt))