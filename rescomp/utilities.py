# -*- coding: utf-8 -*-
""" Implementing various utility functions for the esn

@author: herteux, edited slightly by baur
"""

import numpy as np
import pickle
import time
import datetime
# import scipy.sparse
# import scipy.sparse.linalg
# import matplotlib.pyplot as plt
# import networkx as nx
# import sys
from . import simulations


def load_data(reservoir, data_input=None, mode='data_from_array', starting_point=None,
              add_noise=False, std_noise=0., print_switch=False):
    """
    Method to load data from a file or function (depending on mode)
    for training and testing the network.
    If the file does not exist, it is created according to the parameters.
    reservoir.W_in is initialized.

    :parameters:
    :mode:
    - 'start_from_attractor' uses lorenz.record_trajectory for
        generating a timeseries, by randomly picking starting points from a
        given trajectory.
    - 'fix_start' passes        for t in np.arange(reservoir.discard_steps):
        reservoir.r[0] = np.tanh(
            reservoir.input_weight *
            np.matmul(reservoir.W_in, reservoir.x_discard[t]) + \
            np.matmul(reservoir.network, reservoir.r[0]) ) starting_point to lorenz.record_trajectory
    - 'data_from_array' loads a timeseries from array without further
    checking for reasonability - use with care! For a d-dimensional time series
    with T timesteps use shape (T,d).

    :add_noise: boolean: if normal distributed noise should be added to the
        imported timeseries
    :std_noise: standard deviation of the applied noise
    """
    # pdb.set_trace()
    # print(reservoir)

    t0 = time.time()

    # minimum size for vals, lorenz.save_trajectory has to evaluate:
    timesteps = 1 + reservoir.discard_steps + reservoir.training_steps \
                + reservoir.prediction_steps

    if print_switch:
        print('mode: ', mode)

    if mode == 'data_from_array':

        #            check for dimension and timesteps
        #            reservoir.discard_steps + reservoir.training_steps + \
        #            reservoir.prediction_steps (+1) == vals.shape

        vals = data_input

        # vals -= vals.meactivation_funcan(axis=0)
        # vals *= 1 / vals.std(axis=0)
        # print(vals.shape)

    elif mode == 'start_from_attractor':
        length = 50000
        original_start = np.array([-2.00384153, -5.34877257, -1.20401106])
        random_index = np.random.choice(np.arange(10000, length, 1))
        if print_switch:
            print('random index for starting_point: ', random_index)

        starting_point = simulations.record_trajectory(
            sys_flag=reservoir.sys_flag,
            dt=reservoir.dt,
            timesteps=length,
            starting_point=original_start,
            print_switch=print_switch)[random_index]

        vals = simulations.record_trajectory(sys_flag=reservoir.sys_flag,
                                             dt=reservoir.dt,
                                             timesteps=timesteps,
                                             starting_point=starting_point,
                                             print_switch=print_switch)

    elif mode == 'fix_start':
        if starting_point is None:
            raise Exception('set starting_point to use fix_start')
        else:
            vals = simulations.record_trajectory(sys_flag=reservoir.sys_flag,
                                                 dt=reservoir.dt,
                                                 timesteps=timesteps,
                                                 starting_point=starting_point,
                                                 print_switch=print_switch)
    else:
        raise Exception(mode, ' mode not recognized')
        # print(mode, ' mode not recognized')
    # print('data loading successfull')
    if add_noise:
        vals += np.random.normal(scale=std_noise, size=vals.shape)
        print('added noise with std_dev: ' + str(std_noise))

    #normalization of time series to zero mean and unit std for each
    #dimension individually:
    if reservoir.normalize_data:
        vals -= vals.mean(axis=0)
        vals *= 1/vals.std(axis=0)

    # define local variables for test/train split:
    n_test = reservoir.prediction_steps
    n_train = reservoir.training_steps + reservoir.discard_steps
    n_discard = reservoir.discard_steps

    # sketch of test/train split:
    # [--n_discard --|--n_train--|--n_test--]
    # and y_train is shifted by 1
    reservoir.x_train = vals[n_discard:n_train, :]  # input values driving reservoir
    reservoir.x_discard = vals[:n_discard, :]
    reservoir.y_train = vals[n_discard + 1:n_train + 1, :]  # +1
    reservoir.y_test = vals[n_train + 1:n_train + n_test + 1, :]  # +1

    # check, if y_test has prediction_steps:
    # should be extended to a real test_func!
    if reservoir.y_test.shape[0] != reservoir.prediction_steps:
        print('length(y_test) [' + str(reservoir.y_test.shape[0])
              + '] != prediction_steps [' + str(reservoir.prediction_steps) + ']')

    t1 = time.time()
    if print_switch:
        print('input (x) and target (y) loaded in ', t1 - t0, 's')


def save_realization(reservoir, filename='parameter/test_pickle_'):
    """
    Saves the network parameters (extracted with reservoir.__dict__ to a file,
    for later reuse.

    pickle protocol version 2
    """
    reservoir.timestamp = datetime.datetime.now()
    f = open(filename, 'wb')
    try:
        pickle.dump(reservoir.__dict__, f, protocol=2)
    except:
        print('file could not be pickled: ', filename)
    f.close()

    return 'file saved'


def load_realization(reservoir, filename='parameter/test_pickle_', print_switch=False):
    """
    loads __dict__ from a pickled file and overwrites reservoir.__dict__
    Original data lost!
    """
    g = open(filename, 'rb')
    try:
        dict_load = pickle.load(g)

        keys_init = reservoir.__dict__.keys()
        keys_load = dict_load.keys()

        key_load_list = []
        key_init_list = []
        for key in keys_load:
            reservoir.__dict__[key] = dict_load[key]  # this is where the dict is loaded
            # print(str(key)+' was loaded to __dict__')
            if not key in keys_init:
                key_load_list.append(key)
        if print_switch:
            print('not in initial reservoir: ' + str(key_load_list))
        for key in keys_init:
            if not key in keys_load:
                key_init_list.append(key)
        if print_switch:
            print('not in loaded reservoir: ' + str(key_init_list))

        return 'file loaded, __dict__ available in reservoir.dict'
    except:
        print('file could not be unpickled: ', filename)
    g.close()
