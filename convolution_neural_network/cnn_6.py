"""
convolutional neural network

@author: mabrouk
"""
import numpy as np
import scipy as sci
import networkx as nx 

def cnn(signal, network_dimension, network_edges, network_radius, regularization, 
          q, l, global_dimension, discarding_steps, training_steps, prediction_steps):
    
    N = global_dimension
    signal_dimension = q+2*l
    network = np.asarray(nx.to_numpy_matrix (nx.fast_gnp_random_graph(network_dimension, network_edges)))
    index_nonzero = np.nonzero(network) 
    network[index_nonzero] = np.random.uniform(-1,1, np.count_nonzero(network))
    network = sci.sparse.csr_matrix(network)
    eigenvals = sci.sparse.linalg.eigs(network, 1)[0]
    eigenmax = np.absolute(eigenvals).max()
    network = ((network_radius / eigenmax) * network)
    W_in = np.random.uniform(-1, 1, (network_dimension, signal_dimension))

    network = sci.sparse.block_diag([network for i in range(int(N/q)-2)])
    W_in = sci.sparse.block_diag([W_in for i in range(int(N/q)-2)])
    signal = np.array([np.array([signal[t, i*q-l:(i+1)*q+l] for i in range(1, int(N/q)-1)]).flatten() for t in range(signal.shape[0])])
    r = np.zeros([(int(N/q)-2)*network_dimension])
    
    R = []
    t = 0
    while t < discarding_steps+training_steps-1:
        if t < discarding_steps:
            r = np.tanh(W_in @ signal[t] + network @ r)        
        else:
            r = np.tanh(W_in @ signal[t] + network @ r)
            R.append(r)                        
        t+=1   
    'how to do the inversion without making a loop over all reservoirs ?'
                
    return W_out