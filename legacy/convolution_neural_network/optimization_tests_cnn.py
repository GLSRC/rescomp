# -*- coding: utf-8 -*-
# python 3.7.1
""" Testing the optimizations for the K

author: mabrouk, edited by baur
"""

import numpy as np
import networkx as nx
from numpy.linalg import inv
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
# import numba as nb
from numba import jit, njit
from time import perf_counter
import time


# TODO: Refactor to make more legible, then add a docstring, remove/add print statements etc.
def ks_pde_simulation(dimensions, system_size, t_max, time_step):
    # This function simulates the Kuramoto–Sivashinsky PDE
    # reference for the numerical integration : "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    # https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
    # Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    print("Start PDE simulation")

    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # grid points for the PDE simulation #TODO: I think

    # Define initial conditions and Fourier Transform them
    x = np.transpose(np.conj(np.arange(1, n + 1))) / n
    u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    v = np.fft.fft(u)

    h = time_step  # time step

    # Wave numbers
    k = np.transpose(
        np.conj(np.concatenate((np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0))))) * 2 * np.pi / size

    # Just copied from the paper, it works
    L = k ** 2 - k ** 4
    E = np.exp(h * L)
    E_2 = np.exp(h * L / 2)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

    uu = [np.array(u)]  # List of Real space solutions, later converted to a np.array
    tmax = t_max  # Length of time to simulate
    nmax = round(tmax / h)  # No. of time steps to simulate

    g = -0.5j * k  # TODO: Meaning?

    # See paper for details
    for n in range(1, nmax + 1):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E_2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E_2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E_2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        u = np.real(np.fft.ifft(v))
        uu.append(np.array(u))

    uu = np.array(uu)
    print("PDE simulation finished")

    return uu


# The old CNN function. Used to compare results
def cnn_2(signal, network_dimension, network_edges, network_radius, regularization, q,
          l, global_dimension, discarding_steps, training_steps, prediction_steps, network_file_path=None):
    def discarding(signal, r):
        for t in range(discarding_steps + 1):
            r[0] = np.tanh(W_in @ signal[t] + network @ r[0])
        return r

    def training(signal, r):
        for t in range(training_steps - 1):
            r[t + 1] = np.tanh(W_in @ signal[discarding_steps + t + 1] + network @ r[t])
        return r

    def fitting(signal, r):
        W = (signal[discarding_steps + 1:discarding_steps + training_steps + 1]).T @ r @ \
            inv(r.T @ r + regularization * np.eye(r.shape[1]))
        return W

    N = global_dimension
    signal_dimension = q + 2 * l

    # TODO: This should be a try - except, not an if
    # TODO the network generation/loading/saving should be it's own function actually and everything should be in it's
    #  own a class so that one doesn't have to use 20 function inputs
    if network_file_path != None:
        network = np.load(network_file_path)
    else:
        network = np.asarray(
            nx.to_numpy_matrix(nx.fast_gnp_random_graph(network_dimension, network_edges, seed=np.random)))
        index_nonzero = np.nonzero(network)
        network[index_nonzero] = np.random.uniform(-1, 1, np.count_nonzero(network))
        network = ((network_radius / np.absolute(np.linalg.eigvals(network)).max()) * network)

    # network = np.asarray(nx.to_numpy_matrix(nx.fast_gnp_random_graph(network_dimension, network_edges, seed=np.random)))
    # index_nonzero = np.nonzero(network)
    # network[index_nonzero] = np.random.uniform(-1, 1, np.count_nonzero(network))
    # network = (network_radius / np.absolute(np.linalg.eigvals(network)).max()) * network

    W_in = np.random.uniform(-0.5, 0.5, (network_dimension, signal_dimension))

    r = np.zeros([int(N / q) - 2, training_steps, network_dimension])
    r = np.array([discarding(signal[:, i * q - l:(i + 1) * q + l], r[i - 1]) for i in range(1, int(N / q) - 1)])
    r = np.array([training(signal[:, i * q - l:(i + 1) * q + l], r[i - 1]) for i in range(1, int(N / q) - 1)])
    W_out = np.array([fitting(signal[:, i * q - l:(i + 1) * q + l], r[i - 1]) for i in range(1, int(N / q) - 1)])
    r = np.array([r[i][-1] for i in range(int(N / q) - 2)])

    signal_prediction = np.zeros([prediction_steps, q * (int(N / q) - 2) + 2 * l])
    signal_prediction[0] = np.concatenate(((W_out[0] @ r[0])[:l],
                                           np.array([(W_out[i] @ r[i])[l:-l] for i in range(int(N / q) - 2)]).ravel(),
                                           (W_out[-1] @ r[-1])[-l:]))
    for t in range(prediction_steps - 1):
        r = np.array([np.tanh(W_in @ signal_prediction[t, i * q:(i + 1) * q + 2 * l] + network @ r[i]) \
                      for i in range(int(N / q) - 2)])
        signal_prediction[t + 1] = np.concatenate(((W_out[0] @ r[0])[:l],
                                                   np.array([(W_out[i] @ r[i])[l:-l] for i in
                                                             range(int(N / q) - 2)]).ravel(),
                                                   (W_out[-1] @ r[-1])[-l:]))  # prediction

    signal_test = signal[discarding_steps + training_steps:discarding_steps + training_steps + prediction_steps]
    return signal_prediction, signal_test


# TODO: Reformatting, then write a Docstring after the signature is finalized
def cnn_scipy_sparse(
        signal, global_dimension, network_dimension, network_edges, network_radius, regularization,
        q, l, discarding_steps, training_steps, prediction_steps, network_file_path=None):
    print("Start the prediction via Convolution Neural Network Reservoir Computing")

    array_data_type = np.float64 # Not really needed, only here for testing

    signal_dimension = q + 2 * l

    # TODO: This should be a try - except, not an if
    # TODO the network generation/loading/saving should be it's own function actually and everything should be in it's
    #  own a class so that one doesn't have to use 20 function inputs
    if network_file_path != None:
        network = np.load(network_file_path)
    else:
        network = np.asarray(
            nx.to_numpy_matrix(nx.fast_gnp_random_graph(network_dimension, network_edges, seed=np.random)))
        index_nonzero = np.nonzero(network)
        network[index_nonzero] = np.random.uniform(-1, 1, np.count_nonzero(network))

        network = scipy.sparse.csr_matrix(network)

        eigenvals = scipy.sparse.linalg.eigs(network, 1)[0]
        max = np.absolute(eigenvals).max()

        network = ((network_radius / max) * network)

        # network = np.asarray(
        #     nx.to_numpy_matrix(nx.fast_gnp_random_graph(network_dimension, network_edges, seed=np.random)))
        # index_nonzero = np.nonzero(network)
        # network[index_nonzero] = np.random.uniform(-1, 1, np.count_nonzero(network))
        # network = ((network_radius / np.absolute(np.linalg.eigvals(network)).max()) * network).astype(array_data_type)
        # network = scipy.sparse.csr_matrix(network)

    w_in = np.random.uniform(-0.5, 0.5, (network_dimension, signal_dimension)).astype(array_data_type)
    network = scipy.sparse.csr_matrix(network)

    local_signal = np.array([signal[:, i * q - l:(i + 1) * q + l] for i in range(1, int(global_dimension / q) - 1)])
    signal_prediction = np.zeros([prediction_steps, q * (int(global_dimension / q) - 2) + 2 * l]).astype(
        array_data_type)

    # @njit #TODO: Numba not yet implemented. Also should be another function of the to be done class
    def training(local_signal):
        r = np.zeros([training_steps, network_dimension]).astype(array_data_type)

        train_start = perf_counter()
        t = 0
        while t <= training_steps + discarding_steps - 1:
            if t <= discarding_steps:
                r[0] = np.tanh(w_in.dot(local_signal[t]) + network.dot(r[0]))
            else:
                r[t - discarding_steps] = np.tanh(w_in.dot(local_signal[t]) + network.dot(r[t - discarding_steps - 1]))
            t += 1
        print("Time needed for the training (and discarding) %.3f seconds " % (perf_counter() - train_start))

        fit_start = perf_counter()
        local_w_out = np.linalg.solve((r.T @ r + regularization * np.eye(r.shape[1])), (
                r.T @ (local_signal[discarding_steps + 1:discarding_steps + training_steps + 1]))).T
        print("Time needed for the fitting %.3f seconds " % (perf_counter() - fit_start))

        return local_w_out, r[-1]

    aux = np.array([training(s) for s in local_signal])
    w_out = np.array([x[0] for x in aux])
    r = np.array([x[1] for x in aux])

    signal_prediction[0] = \
        np.concatenate(((w_out[0].dot(r[0]))[:l],
                        np.array([(w_out[i].dot(r[i]))[l:-l] for i in range(int(global_dimension / q) - 2)]).flatten(),
                        (w_out[-1].dot(r[-1]))[-l:]))

    t2_start = perf_counter()
    t = 0
    while t <= prediction_steps - 2:
        r = np.array([np.tanh(w_in.dot(signal_prediction[t, i * q:(i + 1) * q + 2 * l]) + network.dot(r[i])) for i in
                      range(int(global_dimension / q) - 2)])
        signal_prediction[t + 1] = np.concatenate(
            ((w_out[0].dot(r[0]))[:l],
             np.array([(w_out[i].dot(r[i]))[l:-l] for i in range(int(global_dimension / q) - 2)]).flatten(),
             (w_out[-1].dot(r[-1]))[-l:]))
        t += 1
    print("Time for t2 %.3f seconda " % (perf_counter() - t2_start))

    signal_test = signal[discarding_steps + training_steps:discarding_steps + training_steps + prediction_steps]

    print("Local States RC Prediction Done")
    return signal_prediction, signal_test


if __name__ == "__main__":
    script_start_time = time.time()
    print("Program started at " + time.strftime("%H:%M:%S") + "\n")

    pass  # --- Variables, System Setup

    dim = 300
    grid_points = 300
    t_max = 300
    t_step = 0.05
    network_nodes = 1000
    discarding_steps = 1000
    training_steps = 1000
    prediction_steps = 1000

    # A shortcut to not have to re-calculate the 5000x5000 network every time.
    # TODO: The way the loading is implemented here is suuper hacky
    if network_nodes == 5000:
        network_file_path = "network5000.npy"
    else:
        network_file_path = None

    pass  # --- Simulate the KS PDE

    t_pde_start = perf_counter()
    signal = ks_pde_simulation(dimensions=dim, system_size=grid_points, t_max=t_max, time_step=t_step)
    print("Time needed for the PDE simulation: %.3f seconds \n" % (perf_counter() - t_pde_start))

    pass  # --- Plot the signal

    # plt.imshow(signal.T)
    # # plt.colorbar()
    # plt.show()

    pass  # --- Old Local States Reservoir Computing

    print("Start the old RC")

    np.random.seed(0)  # Needed to create the same network and prediction in general

    t_old_start = perf_counter()
    signal_prediction_old, signal_test_old = \
        cnn_2(signal, network_dimension=network_nodes, network_edges=0.02, network_radius=0.9, regularization=0.01,
              q=10, l=2, global_dimension=dim, discarding_steps=discarding_steps, training_steps=training_steps,
              prediction_steps=prediction_steps, network_file_path=network_file_path)
    print("Time needed for the old CNN: %.3f seconds \n" % (perf_counter() - t_old_start))

    pass  # --- New Local States Reservoir Computing

    np.random.seed(0)  # Needed to create the same network and prediction in general

    t_new_start = perf_counter()
    signal_prediction, signal_test = \
        cnn_scipy_sparse(signal, global_dimension=dim, network_dimension=network_nodes, network_edges=0.02,
                         network_radius=0.9, regularization=0.01, q=10, l=2,
                         discarding_steps=discarding_steps, training_steps=training_steps,
                         prediction_steps=prediction_steps, network_file_path=network_file_path)
    print("Time needed for the new RC: %.3f seconds \n" % (perf_counter() - t_new_start))

    pass  # --- Compare Results, plot the predictions

    # Check if the RC function outputs are the same.
    # The prediction will diverge after a couple hundred time steps due to rounding and KS being chaotic!
    print("Signal Tests are the same is:", np.allclose(signal_test_old, signal_test))
    print("Signal Predictions are the same is:", np.allclose(signal_prediction_old, signal_prediction))

    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.imshow(signal_prediction_old.T)
    ax1.set_title("KS Prediction from the old RC function")

    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.imshow(signal_prediction.T)
    ax2.set_title("KS Prediction from the new RC function")

    plt.show()

    print("Time needed in total: ", time.time() - script_start_time)
