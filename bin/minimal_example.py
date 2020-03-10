# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:54:41 2020

@author: herteux, baur
"""
import pdb

import rescomp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d





#Create some data by simulating the Lorenz63 system
train_sync_steps = 300
train_steps = 3000
pred_steps = 400
simulation_time_steps = train_sync_steps + train_steps + pred_steps + 5000

starting_point = np.array([-14.03020521, -20.88693127, 25.53545])
sim_data = rescomp.simulations.simulate_trajectory(
            sys_flag='mod_lorenz', dt=2e-2, time_steps=simulation_time_steps,
            starting_point=starting_point)
            
            
            
            

# --- ESN Usage Example 1

# Create ESN object. You should use ESNWrapper, which has the full functionality.

esn=rescomp.ESNWrapper()


# Optionally you can set a console- or file-logger to print out Information about
# the process while running. There are different options like 'warning' or 'debug'
# to control the verbosity
esn.set_console_logger('warning')
#esn.set_file_logger('debug', 'esn_log.txt')


# Create the actual network of the ESN. Untill now, the object esn was basically
# empty. Parameters are set in the following methods
esn.create_network(n_rad=0.5)



# A typical and natural way to use an ESN is to first synchronize on one section 
# of a trajectory, train on the next 
# and then start the prediction immediately. To do this use the method 
# esn.train_and_predict()

y_pred, y_test = esn.train_and_predict(x_data=sim_data, 
            train_sync_steps=train_sync_steps, train_steps=train_steps,
            pred_steps=pred_steps, w_in_scale=0.2)
 
# This is the the predicted trajectory           
print(y_pred.shape)  

# If x_data.shape[0] is greater than train_sync_steps + train_steps the rest of
# the data can be used as test set for the prediction. 
# WARNING: f x_data.shape[0] - (train_sync_steps + train_steps) < pred_steps 
# then y_test.shape[0]<y_pred.shape[0].    
print(y_test.shape)           





# --- Plotting the results

ax=plt.axes(projection='3d')

ax.plot(y_test[:, 0], y_test[:, 1],
        y_test[:,2],alpha=0.8,color='Blue',label='test_data')
ax.plot(y_pred[:, 0], y_pred[:, 1],y_pred[:,2],alpha=0.8,color='Orange',
        label='prediction')
        
start=y_pred[0]
ax.plot([start[0]],[start[1]],[start[2]], 'o', label='starting point')

plt.legend()

plt.show()






# --- ESN Usage Example 2


# To start the prediction at any other point, use the methods esn.train() and 
# esn.predict() separately.

# The reservoir should always be synchronized with the data to make sure outputs
# are meaningful. This is handled in esn.train() automatically using 
# x_train[:sync_steps].
esn.train(x_train=sim_data[train_sync_steps + train_steps:], 
          sync_steps=train_sync_steps, w_in_scale=0.2) 

# To predict from any part of the trajectory besides the end of the training
# intervall, the reservoir has to be synchronized again. Specify some intervall
# of synchronization steps:
pred_sync_steps=300

#pdb.set_trace()


# Again the synchronization happens automatically in esn.predict() using 
# x_pred[:sync_steps]. The actual prediction starts from the initial point 
# x_pred[pred_sync_steps+1]
y_pred, y_test = esn.predict(x_pred=sim_data[train_sync_steps + train_steps +
                pred_steps + 700:],
                sync_steps=pred_sync_steps, pred_steps=pred_steps)


# This is the the predicted trajectory           
print(y_pred.shape)  

# If x_pred.shape[0] is greater than sync_steps the rest of
# the data can be used as test set for the prediction. 
# WARNING: f x_pred.shape[0] - sync_steps < pred_steps 
# then y_test.shape[0]<y_pred.shape[0].    
print(y_test.shape)           




# --- Plotting the results
 
ax=plt.axes(projection='3d')

ax.plot(y_test[:, 0], y_test[:, 1],
        y_test[:,2],alpha=0.8,color='Blue',label='test_data')
ax.plot(y_pred[:, 0], y_pred[:, 1],y_pred[:,2],alpha=0.8,color='Orange',
        label='prediction')

start=y_pred[0]
ax.plot([start[0]],[start[1]],[start[2]], 'o', label='starting point')

plt.legend()

plt.show()       

