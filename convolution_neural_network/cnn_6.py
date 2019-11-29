"""
convolutional neural network

"""
import numpy as np
import scipy as sci
import networkx as nx
from numpy import matmul
from numpy.linalg import solve
from time import perf_counter

def cnn(signal, network_dimension, network_edges, network_radius, tikonov, 
        q, l, global_dimension, discarding_steps, training_steps, prediction_steps):
    
    N = global_dimension
    signal_dimension=q+2*l
    network_number = int(N/q)-2
    network = np.asarray(nx.to_numpy_matrix(nx.fast_gnp_random_graph(network_dimension, network_edges)))
    index_nonzero = np.nonzero(network)
    network[index_nonzero] = np.random.uniform(-1, 1, np.count_nonzero(network))
    network = sci.sparse.csr_matrix(network)
    eigenmax = np.absolute(sci.sparse.linalg.eigs(network, 1)[0]).max()
    network = ((network_radius / eigenmax) * network)
    W_in = np.random.uniform(-1, 1, (network_dimension, signal_dimension))
    echo = np.zeros([training_steps, network_dimension, network_number])
    signal = np.array([np.array([signal[t, i*q-l:(i+1)*q+l] for i in range(1, network_number+1)]).T for t in range(discarding_steps+training_steps)])

    t_start = perf_counter()
    t = 0
    while t < discarding_steps + training_steps-1:
        if t < discarding_steps:
            echo[0] = np.tanh(W_in @ signal[t] + network @ echo[0])
        else :
            echo[t+1-discarding_steps] = np.tanh(W_in @ signal[t] + network @ echo[t-discarding_steps])
        t+=1
    t_stop = perf_counter()
    print((t_stop-t_start), 'seconds for training echo')  
    
    t_start = perf_counter()
    signal = signal[discarding_steps:]
    last_echo = echo[-1]
    echo = echo.transpose(2, 0, 1).copy()
    signal = signal.transpose(2, 0, 1).copy()
    t_stop = perf_counter()
    print((t_stop-t_start), 'seconds copy')
    
    t_start = perf_counter()
    W_out = solve(echo.transpose(0, 2, 1) @ echo + tikonov * np.eye(network_dimension), echo.transpose(0, 2, 1) @ signal)
    t_stop = perf_counter()
    print((t_stop-t_start), 'seconds for inversion')

    t_start = perf_counter()
    W_out = W_out.transpose(0, 2, 1).copy()
    t_stop = perf_counter()
    print((t_stop-t_start), 'seconds for copy')

    echo = last_echo
    def convolve(signal):
        signal[:l, 1:] = signal[q:q+l, :-1]
        signal[-l:, :-1] = signal[l:2*l, 1:]
        return signal                                                              
    def deconvolve(signal):
        signal = signal[:, l:-l, :]
        prediction = np.array([np.array(s.transpose().flatten()) for s in signal])
        return prediction

    t_start = perf_counter()
    prediction = np.zeros([prediction_steps, signal_dimension, network_number])
    t = 0
    while t < prediction_steps:
        prediction[t] = np.einsum('ikj,ji->ki', W_out, echo)
        echo = np.tanh(W_in @ convolve(prediction[t]) + network @ echo)
        t+=1
    prediction = deconvolve(prediction)    
    t_stop = perf_counter()
    print((t_stop-t_start), 'seconds for predictive echo')
    
    return prediction