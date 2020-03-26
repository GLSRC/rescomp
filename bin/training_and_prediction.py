# -*- coding: utf-8 -*-
""" Training and prediction explained in more detail

@author: herteux, baur
"""
import pdb

import rescomp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# --- Training and prediction

# As before, create an ESN object:
esn = rescomp.ESNWrapper()

# simulate some data
simulation_time_steps = 20000
starting_point = np.array([-14.03020521, -20.88693127, 25.53545])
sim_data = rescomp.simulations.simulate_trajectory(
    sys_flag='mod_lorenz', dt=2e-2, time_steps=simulation_time_steps,
    starting_point=starting_point)

# and create the internal network
esn.create_network()

# For this example, let's use the same number of time steps for the training as
# in before
train_sync_steps = 300
train_steps = 3000

# Let's cut the input data accordingly
x_train = sim_data[: train_sync_steps + train_steps]

# And plug it in the train() method
esn.train(x_train=sim_data, sync_steps=train_sync_steps)

# Now that the reservoir is trained, we want to predict the system evolution not
# directly after the training data (after 3300) but instead sometime later.
# Let's say we want to look at the evolution after 5000 time steps instead:
x_pred = sim_data[5000:]

# Due to this disconnect, we must synchronize the internal reservoir states with
# the system again before we can use it to predict anything

# Let's synchronize for 300 steps again
pred_sync_steps = 300
# And again predict for 500
pred_steps = 400

# Note that the actual prediction starts AFTER the synchronization, i.e. from
# the initial point x_pred[pred_sync_steps + 1].
y_pred, y_test = esn.predict(x_pred=x_pred, sync_steps=pred_sync_steps,
                             pred_steps=pred_steps)

# --- Plot the first prediction

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(y_test[:, 0], color='blue', label='test_data')
ax1.plot(y_pred[:, 0], color='orange', label='prediction')
ax1.set_title("X-Coordinates of Simulation and Prediction after 5300 time steps")

plt.legend()
plt.show()


# --- Another prediction

# Note that we do not need to re-train our reservoir if we want to predict the
# same system from a different starting point again, we just need to synchronize
# it every time. Let's do another prediction, this time after 10000 time steps:
x_pred = sim_data[10000:]

y_pred2, y_test2 = esn.predict(x_pred=x_pred, sync_steps=pred_sync_steps,
                               pred_steps=pred_steps)

# --- Plot the second prediction

fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(y_test2[:, 0], color='blue', label='test_data')
ax2.plot(y_pred2[:, 0], color='orange', label='prediction')
ax2.set_title("X-Coordinates of Simulation and Prediction after 10300 time steps")

plt.legend()
plt.show()
