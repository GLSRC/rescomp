"""
Created on Thu Oct 17 15:59:49 2019

@author: mabrouk
"""
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rnn_ortho import rnn

signal = np.loadtxt(fname = "signal_lorenz")
signal = np.array(signal)
signal -= np.mean(signal, axis=0)
signal /= np.std(signal, axis=0) 
t1_start = perf_counter()
signal_prediction, signal_test = rnn(signal, signal_scaling=0.3, 
                                     network_dimension=500,  
                                     network_scaling=0.9, tikonov=0.000001, 
                                     block_dimension=10, signal_dimension=3, 
                                     discarding_steps=1000, 
                                     training_steps=10000, 
                                     prediction_steps=2000)
t1_stop = perf_counter()
print((t1_stop-t1_start), "seconds")
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.plot(signal_prediction[:, 0], signal_prediction[:, 1], signal_prediction[:, 2], lw=0.5)
plt.plot(signal_test[:, 0], signal_test[:, 1], signal_test[:, 2], lw=0.5)
plt.show()
plt.plot(signal_prediction[:, 0])
plt.plot(signal_test[:, 0])
plt.show()

signal = np.loadtxt(fname = "signal_roessler")
signal = np.array(signal)
signal -= np.mean(signal, axis=0)
signal /= np.std(signal, axis=0) 
t1_start = perf_counter()
signal_prediction, signal_test = rnn(signal, signal_scaling=0.05, 
                                     network_dimension=500,  
                                     network_scaling=0.5, tikonov=0.000001, 
                                     block_dimension=10, signal_dimension=3, 
                                     discarding_steps=1000, 
                                     training_steps=10000, 
                                     prediction_steps=4000)
t1_stop = perf_counter()
print((t1_stop-t1_start), "seconds")
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.plot(signal_prediction[:, 0], signal_prediction[:, 1], signal_prediction[:, 2], lw=0.5)
plt.plot(signal_test[:, 0], signal_test[:, 1], signal_test[:, 2], lw=0.5)
plt.show()
plt.plot(signal_prediction[:, 0])
plt.plot(signal_test[:, 0])
plt.show()