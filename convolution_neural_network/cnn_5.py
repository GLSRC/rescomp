"""
convolutional neural network

@author: mabrouk
"""
import numpy as np
import scipy as sci
import networkx as nx 
import matplotlib.pyplot as plt

def cnn(signal, network_dimension, network_edges, network_radius, regularization, 
          q, l, global_dimension, discarding_steps, training_steps, prediction_steps):
    
    N = global_dimension
    signal_dimension = q+2*l
    network = np.asarray(nx.to_numpy_matrix(nx.fast_gnp_random_graph(network_dimension, network_edges)))
    index_nonzero = np.nonzero(network) 
    network[index_nonzero] = np.random.uniform(-1,1, np.count_nonzero(network))
    network = sci.sparse.csr_matrix(network)
    eigenvals = sci.sparse.linalg.eigs(network, 1)[0]
    eigenmax = np.absolute(eigenvals).max()
    network = ((network_radius / eigenmax) * network)
    layer_in = np.random.uniform(-0.5, 0.5, (network_dimension, signal_dimension))

    signal_test = signal[discarding_steps+training_steps:discarding_steps+training_steps+prediction_steps, q-l:l-q]     
    signal = np.array([signal[:, i*q-l:(i+1)*q+l] for i in range(1, int(N/q)-1)])
    layer_out = np.zeros([int(N/q)-2, signal_dimension, network_dimension])
    echo = np.zeros([int(N/q)-2, network_dimension])

    r = np.zeros([training_steps, network_dimension])
    i = 0
    while i < int(N/q)-2:
        r[0] = 0. 
        t = 0
        while t < discarding_steps+training_steps-1:
            if t < discarding_steps:
                r[0] = np.tanh(layer_in @ signal[i][t] + network @ r[0])        
            else:
                r[t-discarding_steps+1] = np.tanh(layer_in @ signal[i][t] + network @ r[t-discarding_steps])
            t+=1   
        layer_out[i] = np.linalg.solve((r.T @ r + regularization * np.eye(r.shape[1])), (
                r.T @ (signal[i][discarding_steps:discarding_steps+training_steps]))).T
        echo[i] = r[-1]
        i +=1

    signal_prediction = np.zeros([prediction_steps, q*(int(N/q)-2)+2*l])
    signal_prediction[0] = np.concatenate(((layer_out[0] @ echo[0])[:l], \
     np.array([(layer_out[i] @ echo[i])[l:-l] for i in range(int(N/q)-2)]).ravel(), (layer_out[-1] @ echo[-1])[-l:]))
    t = 0
    while t <= prediction_steps-2:
        echo = np.array([np.tanh(layer_in @ signal_prediction[t, i*q:(i+1)*q+2*l] + network @ echo[i])  for i in range(int(N/q)-2)])
        signal_prediction[t+1] = np.concatenate(((layer_out[0] @ echo[0])[:l], \
         np.array([(layer_out[i] @ echo[i])[l:-l] for i in range(int(N/q)-2)]).flatten(), (layer_out[-1] @ echo[-1])[-l:]))
        t+=1
       
    return signal_prediction, signal_test

