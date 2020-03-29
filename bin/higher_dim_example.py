# -*- coding: utf-8 -*-
""" Reservoir Computing example for the higher dimensional KS system """

import numpy as np
import matplotlib.pyplot as plt
import rescomp

# --- Higher dimensional example

# While the previous examples demonstrated the basic usage of the rescomp
# package, one very important part still needs to be discussed: The
# hyperparameters of RC.

# To do so, here we look at a much higher dimensional system than before, the
# Kuramoto Sivashinsky (KS) system, which we scale to be 22 dimensional.

# As always we create an ESN object
esn = rescomp.ESNWrapper()

# and simulate the system:
simulation_time_steps = 60000
sim_data = rescomp.simulations.simulate_trajectory(
    sys_flag='kuramoto_sivashinsky', dimensions=100, system_size=22, dt=0.05,
    time_steps=simulation_time_steps)

# As before, we want to train the system with data sampled from the chaotic
# attractor we want to predict. As a result, we have to throw the first 1000
# simulated data points away, as they are not on said attractor, but instead
# correspond to transient system dynamics of the KS system.
sim_data = sim_data[1000:]

# Now we again want to create the reservoir/network, but this time the default
# parameters are not sufficient for RC to learn the system dynamics. For the KS
# system the following parameters work:

# The network dimension n_dim. The more complicated and higher dimensional the
# system is, the larger the network needs to be.
n_dim = 5000
# The spectral radius n_rad should be smaller than 1 and must be larger than 0
n_rad = 0.3
# The average network degree n_avg_deg.
n_avg_deg = 100
# These hyperparameters (and the ones below) have been found by trial and error.
# So far no reliable heuristics exist for what the correct hyperparameter ranges
# for an arbitrary system might be. See the FAQ for more details.

# Furthermore, one can set the numpy random seed, to generate the same network
# everytime. This too is a real hyperparameter and different random networks
# vary hugely in their prediction performance.
np.random.seed(0)

# Finally, create the network with those parameters.
esn.create_network(n_dim=n_dim, n_rad=n_rad, n_avg_deg=n_avg_deg)

# The train/train_and_predict methods too, have a set of hyperparameters one
# needs to optimize. The most important of which are the regularization
# parameter which is typically somewhere between 1e-2 and 1e-6
reg_param = 1e-2
# and the scale of the random elements in the input matrix w_in which is usually
# between 0.1 and 1.0
w_in_scale = 1.0

# Define the number of training/prediction time steps
train_sync_steps = 1000
train_steps = 50000
pred_steps = 500

# And train+predict the system
y_pred, y_test = esn.train_and_predict(
    sim_data, train_sync_steps=train_sync_steps, train_steps=train_steps,
    pred_steps=pred_steps, reg_param=reg_param, w_in_scale=w_in_scale)

# --- Plot the results

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(3, 1, 1)
ax1.imshow(y_test.T)
ax1.set_title("Simulation")

ax2 = fig1.add_subplot(3, 1, 2)
ax2.imshow(y_pred.T)
ax2.set_title("Prediction")

ax3 = fig1.add_subplot(3, 1, 3)
ax3.imshow(y_pred.T - y_test.T)
ax3.set_title("Difference between simulation and prediction")

plt.show()
