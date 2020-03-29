# -*- coding: utf-8 -*-
""" Minimal usage example """

import rescomp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# --- Minimal Example

# Most of the functionality of this package is bundled in two main classes: ESN
# and ESNWrapper. (ESN stands for for Echo State Network)
# For all normal usage of the rescomp package, the child class ESNWrapper is the
# class to use.

esn = rescomp.ESNWrapper()

# As with most machine learning techniques, we need some training data, which we
# generate here artificially by simulating the chaotic Lorenz63 system for 20000
# time steps
simulation_time_steps = 20000

# The starting point is chosen to be on the chaotic attractor we want to predict
starting_point = np.array([-14.03020521, -20.88693127, 25.53545])

sim_data = rescomp.simulations.simulate_trajectory(
    sys_flag='mod_lorenz', dt=2e-2, time_steps=simulation_time_steps,
    starting_point=starting_point)

# As a convention, the data specified is always input data in an np.ndarray of
# the shape (T, d), where T is the number of time steps and d the system
# dimension
# I.g. RC works for any number of dimensions and data length.

# Until now, the object esn is basically empty. One main component of reservoir
# computing is the reservoir, i.e. the internal network, wich we can create via
# create_network()
esn.create_network()

# A typical and natural way to use an ESN is to first synchronize on one section 
# of a trajectory, train on the next and then start the prediction on the
# subsequent data
# The method train_and_predict() does exactly that.

# The number of time steps used to synchronize the reservoir, should be at least
# a couple hundred, but no more than a couple thousand are needed, even for
# complex systems.
train_sync_steps = 300
# The number of time steps used to train the reservoir. This depends hugely on
# the system in question. See the FAQ for more details
train_steps = 3000
# The number of time steps predicted.
pred_steps = 400

# Plug it all in
y_pred, y_test = esn.train_and_predict(x_data=sim_data,
                                       train_sync_steps=train_sync_steps,
                                       train_steps=train_steps,
                                       pred_steps=pred_steps)

# The first output y_pred is the the predicted trajectory
print(y_pred.shape)

# If the input data is longer than the data used for synchronization and
# training, i.e. if
#   x_data.shape[0] > train_sync_steps + train_steps,
# then the rest of the data can be used as test set to compare the prediction
# against. If the prediction where perfect y_pred and y_test would be the same
# Be careful though, if
#   x_data.shape[0] - (train_sync_steps + train_steps) < pred_steps
# then the test data set is shorter than the predicted one:
#   y_test.shape[0] < y_pred.shape[0].
print(y_test.shape)

# --- Plot the prediction

ax = plt.axes(projection='3d')

ax.plot(y_test[:, 0], y_test[:, 1], y_test[:, 2],
        alpha=0.8, color='blue', label='test_data')
ax.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2],
        alpha=0.8, color='orange', label='prediction')

start = y_pred[0]
ax.plot([start[0]], [start[1]], [start[2]], 'o', label='starting point')

plt.legend()
plt.show()

# --- Plot the x coordinates as a comparison

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(y_test[:, 0], color='blue', label='test_data')
ax1.plot(y_pred[:, 0], color='orange', label='prediction')
ax1.set_title("X-Coordinates of Simulation and Prediction")

plt.legend()
plt.show()
