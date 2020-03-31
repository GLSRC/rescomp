# -*- coding: utf-8 -*-
""" Usage example for the rescomp.simulations module"""

import rescomp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# --- Simulate the normal Lorenz 63 System

simulation_time_steps = 1000
starting_point = np.array([-14.03020521, -20.88693127, 25.53545])

sim_data = rescomp.simulate_trajectory(
    sys_flag='lorenz', dt=2e-2, time_steps=simulation_time_steps,
    starting_point=starting_point)


# --- Plot the normal Lorenz 63 System

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sim_data[:, 0], sim_data[:, 1], sim_data[:, 2],
        alpha=0.8, color='blue', label='normal Lorenz63 system')
plt.legend()
plt.show()


# --- Simulate the Lorenz 63 System with parameters different than the default
sigma=20  # default: sigma=10
rho=14  # default: rho=28
b=8/3  # default: b=8/3
# For more information on the possible parameters, please see the HTML
# documentation on the simulate_trajectory function

sim_data = rescomp.simulate_trajectory(
    sys_flag='lorenz', dt=2e-2, time_steps=simulation_time_steps,
    starting_point=starting_point, sigma=sigma, rho=rho, b=b)


# --- Plot the Lorenz 63 with different parameters

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sim_data[:, 0], sim_data[:, 1], sim_data[:, 2],
        alpha=0.8, color='blue', label='Lorenz63 with different parameters')
plt.legend()
plt.show()


# --- To simulate the Roessler System instead, just change the sys_flag parameter

simulation_time_steps = 5000
starting_point = np.array([0, 0, 0])

sim_data = rescomp.simulate_trajectory(
    sys_flag='roessler', dt=2e-2, time_steps=simulation_time_steps,
    starting_point=starting_point)

# --- Plot the Roessler System

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sim_data[:, 0], sim_data[:, 1], sim_data[:, 2],
        alpha=0.8, color='blue', label='Roessler System')
plt.legend()
plt.show()

