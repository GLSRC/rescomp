"""
Created on Fri Dec 13 16:09:13 2019
@author: mabrouk
"""
import numpy as np
import scipy as sci
import networkx as nx
from joblib import Parallel, delayed

def mean_rnn(signal, network_dimension, network_density, 
             network_scaling, signal_scaling, tikonov, 
             signal_dimension, discarding_steps, training_steps, 
             prediction_steps, network_number): 
                 
    layer_in = np.zeros((network_dimension, signal_dimension))            
    for i in range(network_dimension):
        random_coord = np.random.choice(np.arange(signal_dimension))
        layer_in[i, random_coord] = np.random.uniform(-signal_scaling, \
                                                        signal_scaling)     
    bin_network = nx.to_numpy_matrix(nx.fast_gnp_random_graph \
                            (network_dimension, network_density)) 
    def sampling():
        network = bin_network
        network[np.nonzero(network)] = np.random.uniform(0,1, \
                                        np.count_nonzero(network))
        network = sci.sparse.csr_matrix(network)
        eigenvals = sci.sparse.linalg.eigs(network, 1)[0]
        eigenmax = np.absolute(eigenvals).max()
        network = (network_scaling / eigenmax)*network
        return network
    network = [sampling() for i in range(network_number)]                     
    
    def training(i):
        echo = np.zeros([training_steps, network_dimension])    
        t = 0
        while t < discarding_steps+training_steps-1:
            if t < discarding_steps:
                echo[0] = np.tanh(layer_in@signal[t]+network[i]@echo[0])        
            else:
                echo[t-discarding_steps+1] = np.tanh(layer_in@signal[t]+\
                                       network[i]@echo[t-discarding_steps])   
            t+=1
        R = np.block([echo, echo**2])
        layer_out = np.linalg.solve(R.T@R+tikonov*np.eye(2*network_dimension), 
          R.T @ signal[discarding_steps:discarding_steps+training_steps, :]).T
        return echo[-1], layer_out   
    x = Parallel(n_jobs=5)(delayed(training)(i) for i in range(network_number))
    echos = [x[i][0] for i in range(network_number)]
    layers_out = [x[i][1] for i in range(network_number)]
   
    def prediction(i):
        echo = echos[i]
        signal_prediction = np.zeros([prediction_steps, signal_dimension])
        signal_prediction[0] = layers_out[i] @ \
          np.concatenate((echo, echo**2), axis=0)
        t = 0
        while t < prediction_steps-1:
            echo = np.tanh(layer_in@signal_prediction[t]+network[i]@echo)
            signal_prediction[t+1] = layers_out[i] @ \
               np.concatenate((echo, echo**2), axis=0)
            t+=1  
        return signal_prediction
    signal_prediction = [prediction(i) for i in range(network_number)]
    signal_test = signal[discarding_steps+training_steps-1: \
                        discarding_steps+training_steps+prediction_steps-1, :]
                        
    def model_accuracy(i):
        squared_error = (signal_test-signal_prediction[i])**2
        nrmse = np.sqrt(squared_error.mean(1)/(signal_test**2).mean(1)) 
        return nrmse
    nrmse = [model_accuracy(i) for i in range(network_number)]
    threshold = 0.1
    def prediction_time(normalized_error):
        return np.argmax(normalized_error > threshold)/100
    prediction_times = [prediction_time(error) for error in nrmse] 
    mean_prediction_time = np.array(prediction_times).mean()                   
    return mean_prediction_time