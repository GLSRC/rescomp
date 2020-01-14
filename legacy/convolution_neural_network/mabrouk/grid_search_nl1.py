"""
Created on Thu Jan  9 14:28:37 2020
@author: mabrouk
"""
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial
from rnn_ortho import rnn
from time import perf_counter

signal = np.loadtxt(fname = "lorenz_signal")
signal = np.array(signal)
signal -= np.mean(signal, axis=0)
signal /= np.std(signal, axis=0)
r_min = 0.5
r_max = 2.5
r_steps = 0.1 
radii = np.arange(r_min, r_max, r_steps)
def metric(signal_scaling, network_scaling):
    prediction, test = rnn(signal, signal_scaling=signal_scaling, 
                           network_dimension=500,  
                           network_scaling=network_scaling, 
                           tikonov=0.000001, 
                           block_dimension=10, signal_dimension=3, 
                           discarding_steps=1000, 
                           training_steps=10000, 
                           prediction_steps=2000)                    
    nr_points = float(prediction.shape[0])
    tree = scipy.spatial.cKDTree(prediction)
    N_r = np.array(tree.count_neighbors(tree, radii), dtype=float)/nr_points
    N_r = np.vstack((radii, N_r))
    slope, intercept = np.polyfit(np.log(N_r[0]), np.log(N_r[1]), deg=1)
    return np.sqrt((slope-1.72)**2)          
sample_number = 500
signal_scaling_origin = 0.4
network_scaling_origin = 0.4
signal_scaling_range = 0.4
network_scaling_range = 0.4
def signal_scaling(i):
   return signal_scaling_origin + (2*i/sample_number-1)*signal_scaling_range
def network_scaling(j):
    return network_scaling_origin + (2*j/sample_number-1)*network_scaling_range 
t_start = perf_counter()
energy_landscape = [Parallel(n_jobs=10)\
                (delayed(metric)(signal_scaling(i), network_scaling(j)) \
                 for i in range(sample_number)) for j in range(sample_number)]
t_end = perf_counter()
print((t_end-t_start)/60, "minutes")
energy_landscape = np.array(energy_landscape)
np.savetxt('lorenz_new_landscape', energy_landscape)
X, Y = np.meshgrid(range(sample_number), range(sample_number))
X = signal_scaling_origin + (2*X/sample_number-1)*signal_scaling_range
Y = network_scaling_origin + (2*Y/sample_number-1)*network_scaling_range                    
plt.imshow(energy_landscape)
plt.colorbar()
plt.show()
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d', \
      xlabel="signal_scaling", ylabel="network_scaling") 
ha.plot_surface(X, Y, energy_landscape)
plt.show()