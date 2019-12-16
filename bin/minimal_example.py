# -*- coding: utf-8 -*-
""" Minimal Reservoir Computing example, simulating and then predicting the Lorenz63 system

@author: herteux, edited slightly by baur
"""
import numpy as np
import matplotlib.pyplot as plt

from rescomp import esn
from rescomp.simulations import record_trajectory

# Create some data
trajectory = record_trajectory(sys_flag='mod_lorenz', dt=2e-2, timesteps=int(5e4),
                print_switch=False,
                starting_point=np.array([-2.00384153, -5.34877257, -1.20401106]))[int(-1e4):]

data = trajectory

# Example for using esn
"""
First create an esn object. You can specify the number of nodes you want to 
use (default value=500). At the moment it is necessary to explicitly state the 
dimension of the input and output (default value=3) as well as the number of 
time steps used for training, prediction(testing) and the number of discarded
steps. Since this is very inconvenient it will be changed asap.
The discarded steps are needed to synchronize the reservoir with the input and
are not used for training or prediction. 
"""

e = esn(network_dimension=500, input_dimension=data.shape[1],
        output_dimension=data.shape[1], training_steps=4000,
        prediction_steps=4000, discard_steps=1999)

"""
Specify the data you want to use for training and testing with esn.load_data()
For a d-dimensional time series with T steps the argument should have the shape
(T,d). Currently, this function divides the data into discarded steps, training
data and test data automatically. The first part of the data will be discarded,
the second part for training and the last part (beginning from time step 
discard_steps + training_steps + 1) will be used as test data. 
"""
e.load_data(data)

"""
Synchronizes the reservoir and train on the training data.
"""
e.train()

"""
This creates a prediction of the test data set and saves it in esn.y_pred.
"""
e.predict()


# Plotting the results
"""
You can access the predicted data in esn.y_pred, the test data in esn.y_test,
the training data in esn.x_train and esn.y_train respectively (these are just 
shifted by one time step) and the discarded time steps in esn.x_discard.
"""
pred = e.y_pred
test = e.y_test

plt.plot(test[:, 0], test[:, 1])
plt.plot(pred[:, 0], pred[:, 1])

plt.show()
