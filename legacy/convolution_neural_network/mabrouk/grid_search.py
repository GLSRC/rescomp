"""
Created on Thu Jan  9 14:28:37 2020
@author: mabrouk
"""
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rnn_ortho import rnn
from time import perf_counter

signal = np.loadtxt(fname = "signal_lorenz")
signal = np.array(signal)
signal -= np.mean(signal, axis=0)
signal /= np.std(signal, axis=0) 
def metric(signal_scaling, network_scaling):
    test, prediction = rnn(signal, signal_scaling=signal_scaling, 
                           network_dimension=500,  
                           network_scaling=network_scaling, 
                           tikonov=0.000001, 
                           block_dimension=10, signal_dimension=3, 
                           discarding_steps=1000, 
                           training_steps=10000, 
                           prediction_steps=5000)                       
    first_moment = np.sqrt((np.mean(test, axis=0)-np.mean(prediction, \
      axis=0))**2).mean()
    second_moment = np.sqrt((np.std(test, axis=0)-np.std(prediction, \
      axis=0))**2).mean()
    return first_moment, second_moment           
sample_number = 50
signal_scaling_origin = 0.3
network_scaling_origin = 0.9
signal_scaling_range = 0.1
network_scaling_range = 0.1
def signal_scaling(i):
    return signal_scaling_origin + (2*i/sample_number-1)*signal_scaling_range
def network_scaling(j):
    return network_scaling_origin + (2*j/sample_number-1)*network_scaling_range 
t_start = perf_counter()
energy_landscape = [Parallel(n_jobs=6)\
                (delayed(metric)(signal_scaling(i), network_scaling(j)) \
                 for i in range(sample_number)) for j in range(sample_number)]
t_end = perf_counter()
print((t_end-t_start)/60, "minutes")
mean_landscape = np.zeros([sample_number, sample_number])
variance_landscape = np.zeros([sample_number, sample_number])
for i in range(sample_number):
    for j in range(sample_number):
        mean_landscape[i, j] = energy_landscape[i][j][0]
        variance_landscape[i, j] = energy_landscape[i][j][1] 
                     
plt.imshow(mean_landscape, \
   extent=[signal_scaling(0), signal_scaling(sample_number), \
     network_scaling(0), network_scaling(sample_number)])
plt.colorbar()
plt.show()
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d', \
      xlabel="signal_scaling", ylabel="network_scaling")
X, Y = np.meshgrid(range(sample_number), range(sample_number))
X = signal_scaling_origin + (2*X/sample_number-1)*signal_scaling_range
Y = network_scaling_origin + (2*Y/sample_number-1)*network_scaling_range 
ha.plot_surface(X, Y, mean_landscape)
plt.show()
plt.imshow(variance_landscape, \
   extent=[signal_scaling(0), signal_scaling(sample_number), \
     network_scaling(0), network_scaling(sample_number)])
plt.colorbar()
plt.show()
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d', \
      xlabel="signal_scaling", ylabel="network_scaling")
X, Y = np.meshgrid(range(sample_number), range(sample_number))
X = signal_scaling_origin + (2*X/sample_number-1)*signal_scaling_range
Y = network_scaling_origin + (2*Y/sample_number-1)*network_scaling_range  
ha.plot_surface(X, Y, variance_landscape)
plt.show()