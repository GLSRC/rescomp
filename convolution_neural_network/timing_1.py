# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:59:49 2019

@author: mabrouk
"""
import numpy as np
import matplotlib.pyplot as plt
from cnn_5 import cnn
from time import perf_counter

signal = np.loadtxt(fname = "signal_2")
signal = np.array(signal)[:, 7:]       
print(signal.shape)

t1_start = perf_counter()
signal_prediction, signal_test = cnn(signal, network_dimension=5000, 
                                     network_edges=0.002, 
                                     network_radius=0.6, 
                                     regularization=0.1, q=8, l=6, 
                                     global_dimension=504, 
                                     discarding_steps=1000, 
                                     training_steps=30000, 
                                     prediction_steps=500)
t1_stop = perf_counter()
print((t1_stop-t1_start)/60, "minutes")


plt.figure()
plt.subplot(211)
plt.imshow(signal_test.T, cmap='jet', aspect='auto')
plt.subplot(212)
plt.imshow(signal_prediction.T,  cmap='jet', aspect='auto')
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(signal_test.T, cmap='jet', aspect='auto')
plt.show()
plt.figure(figsize=(10,10))
plt.imshow(signal_prediction.T, cmap='jet', aspect='auto')
plt.show()

error = (signal_test-signal_prediction)**2
plt.figure()
plt.subplot(211)
plt.imshow(np.sqrt(error.T))
plt.subplot(212)
plt.plot(np.sqrt(error.mean(1)/(signal_test**2).mean(1))[:150])
plt.show()

