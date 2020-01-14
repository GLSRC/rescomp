"""
Created on Thu Jan  9 13:03:55 2020
@author: mabrouk
"""
import numpy as np
import scipy.sparse

def rnn(signal, signal_scaling, network_dimension,  
        network_scaling, tikonov, block_dimension,
        signal_dimension, discarding_steps, training_steps, 
        prediction_steps):
    
    network = np.zeros([network_dimension, network_dimension])
    for i in range(1, network_dimension):
        network[i, i-1] = 1.0
    for i in range(1, int(network_dimension/block_dimension)):
        network[i*block_dimension, i*block_dimension-1] = 0
    network *= network_scaling
    network = scipy.sparse.csr_matrix(network)
    
    layer_in = np.zeros((network_dimension, signal_dimension))
    block1 = np.zeros([block_dimension, signal_dimension])
    block1[0, 0] = 1.0
    block1[block_dimension-4:block_dimension-2, 1] = 1.0
    block1[block_dimension-2:, 2]
    block2 = np.zeros([block_dimension, signal_dimension])
    block2[0, 1] = 1.0
    block2[block_dimension-4:block_dimension-2, 0] = 1.0
    block2[block_dimension-2:, 2] = 1.0
    block3 = np.zeros([block_dimension, signal_dimension])
    block3[0, 2] = 1.0
    block3[block_dimension-4:block_dimension-2, 0] = 1.0
    block3[block_dimension-2:, 1] = 1.0
    subblock = np.vstack([block1, block2, block3])
    for i in range(int(network_dimension/(block_dimension*signal_dimension))):
        layer_in[i*(signal_dimension*block_dimension): \
            (i+1)*(signal_dimension*block_dimension)] = subblock
    for i in range(network_dimension):
        layer_in[i, :] *= i/network_dimension
    layer_in *= network_scaling 
    layer_in = scipy.sparse.csr_matrix(layer_in)            
    
    echo = np.zeros([training_steps, network_dimension])    
    t = 0
    while t < discarding_steps+training_steps-1:
        if t < discarding_steps:
            echo[0] = np.tanh(layer_in@signal[t]+network@echo[0])        
        else:
            echo[t-discarding_steps+1] = np.tanh(layer_in@signal[t]+\
                network@echo[t-discarding_steps])   
        t+=1
    R = np.block([echo, echo**2])
    layer_out = np.linalg.solve(R.T@R+tikonov*np.eye(2*network_dimension), 
          R.T @ signal[discarding_steps:discarding_steps+training_steps, :]).T
    echo = echo[-1]
   
    signal_prediction = np.zeros([prediction_steps, signal_dimension])
    signal_prediction[0] = layer_out @ np.concatenate((echo, echo**2), axis=0)
    t = 0
    while t < prediction_steps-1:
        echo = np.tanh(layer_in@signal_prediction[t]+network@echo)
        signal_prediction[t+1] = layer_out @ \
        np.concatenate((echo, echo**2), axis=0)
        t+=1  
    signal_test = signal[discarding_steps+training_steps-1: \
                        discarding_steps+training_steps+prediction_steps-1, :]
    return signal_prediction, signal_test