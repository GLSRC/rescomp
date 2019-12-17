"""
Created on Mon Dec  9 12:56:13 2019
@author: mabrouk
"""

import numpy as np
import scipy as sci
import networkx as nx 

def cnn(signal, network_dimension, network_edges, network_radius, regularization, 
          q, l, global_dimension, discarding_steps, training_steps, prediction_steps):
    
    N = global_dimension
    signal_dimension = q+2*l
    network_number = int(N/q)
    network = nx.to_numpy_matrix(nx.fast_gnp_random_graph(network_dimension, network_edges))
    index_nonzero = np.nonzero(network) 
    network[index_nonzero] = np.random.uniform(0,1, np.count_nonzero(network))
    network = sci.sparse.csr_matrix(network)
    eigenvals = sci.sparse.linalg.eigs(network, 1)[0]
    eigenmax = np.absolute(eigenvals).max()
    network = (network_radius / eigenmax)*network
    layer_in = np.zeros([network_dimension, signal_dimension])
    x = np.arange(network_dimension)
    for i in range(signal_dimension):
        y = np.random.choice(x, int(network_dimension/signal_dimension))
        x = [z for z in x if z not in y]
        for j in y:
            layer_in[j, i] = np.random.uniform(-1, 1)              

    signal_test = signal[discarding_steps+training_steps-1: \
                        discarding_steps+training_steps+prediction_steps-1, :] 
    signal = np.array([np.block([signal[:, -l:], signal[:, :q+l]]), \
           *[signal[:, i*q-l:(i+1)*q+l] for i in range(1, network_number-1)],\
              np.block([signal[:, -(q+l):], signal[:, :l]])])       
    layer_out = np.zeros([network_number, signal_dimension-2*l, network_dimension])
    echo = np.zeros([network_number, network_dimension])
            
    i = 0
    while i < network_number:
        r = np.zeros([training_steps, network_dimension]) 
        t = 0
        while t < discarding_steps+training_steps-1:
            if t < discarding_steps:
                r[0] = np.tanh(layer_in@signal[i][t]+network@r[0])        
            else:
                r[t-discarding_steps+1] = np.tanh(layer_in@signal[i][t]+network@r[t-discarding_steps])   
            t+=1   
        echo[i] = r[-1]
        r[:, 1::2] = r[:, 1::2]**2
        layer_out[i] = np.linalg.solve((r.T@r+regularization*np.eye(network_dimension)), 
            (r.T @ (signal[i][discarding_steps:discarding_steps+training_steps, l:q+l]))).T  
        print(i+1, "th reservoir done")
        i +=1 
        
    signal_prediction = np.zeros([prediction_steps, global_dimension])
    echo_temp = echo
    echo_temp[:, 1::2] = echo_temp[:, 1::2]**2
    signal_prediction[0] = np.array([(layer_out[i] @ echo_temp[i]) for i in range(network_number)]).flatten()
    t = 0
    while t < prediction_steps-1:
        first_signal = np.concatenate((signal_prediction[t, -l:], signal_prediction[t, :q+l]))
        first_echo = np.tanh(layer_in@first_signal + network@echo[0])
        last_signal = np.concatenate((signal_prediction[t, -(q+l):], signal_prediction[t, :l]))
        last_echo = np.tanh(layer_in@last_signal + network@echo[-1])
        center_echo = np.array([np.tanh(layer_in@signal_prediction[t, i*q-l:(i+1)*q+l]+network@echo[i]) for i in range(1, network_number-1)])
        
        echo = np.vstack((first_echo, center_echo, last_echo))
        echo_temp = echo
        echo_temp[:, 1::2] = echo_temp[:, 1::2]**2
        signal_prediction[t+1] = np.array([(layer_out[i] @ echo_temp[i]) for i in range(network_number)]).flatten()
        t+=1      
    return signal_prediction, signal_test