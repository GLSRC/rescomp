# -*- coding: utf-8 -*-
""" Implementing various utility functions for the esn

@author: herteux, edited slightly by baur
"""

import numpy as np
import pickle
import time
import datetime
import logging
import sys
# import scipy.sparse
# import scipy.sparse.linalg
# import matplotlib.pyplot as plt
# import networkx as nx
from . import simulations


class ESNLogging:
    """ Custom logging class, logging both to stdout as well as to file

    Use this as you would use the standard logging module, by calling
    ESNLogging.logger.logLevel(msg) with logLevel being on of [DEBUG, INFO,
    WARNING, ERROR, CRITICAL] and msg a human readable message string

    Inspired by
    https://docs.python.org/3/howto/logging-cookbook.html#using-logging-in-multiple-modules
    and
    https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
    and
    https://stackoverflow.com/a/13733863
    """

    def __init__(self):

        self.logger = None

        self._log_file_path = None
        self._console_log_level = None
        self._file_log_level = None
        self._logger_name = None

        self._file_handler = None
        self._console_handler = None

        self._log_level_synonyms = SynonymDict()
        self._log_level_synonyms.add_synonyms(0, ["NOTSET", "notset"])
        self._log_level_synonyms.add_synonyms(10, ["DEBUG", "debug"])
        self._log_level_synonyms.add_synonyms(20, ["INFO", "info"])
        self._log_level_synonyms.add_synonyms(30, ["WARNING", "warning"])
        self._log_level_synonyms.add_synonyms(40, ["ERROR", "error"])
        self._log_level_synonyms.add_synonyms(50, ["CRITICAL", "critical"])
        self._log_level_synonyms.add_synonyms(100, ["OFF", "off"])

        self._create_logger()
        self.set_console_logger(log_level="debug")
        # self.set_file_logger()

    def set_console_logger(self, log_level):
        """ Set loglevel for the console output

        Args:
            log_level (): console loglevel as specified in:
                https://docs.python.org/3/library/logging.html#logging-levels

        """
        self._console_log_level = self._log_level_synonyms.get_flag(log_level)

        # Remove the old handler if there is one
        if self._console_handler is not None:
            self.logger.removeHandler(self._console_handler)

        if self._console_log_level == 100:
            pass  # deactivated logger
        else:
            # console log output format
            ch_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)-7s] %(message)s',
                datefmt='%m-%d %H:%M:%S')

            # Create new console handler
            self._console_handler = logging.StreamHandler(stream=sys.stdout)
            self._console_handler.setLevel(self._console_log_level)
            self._console_handler.setFormatter(ch_formatter)

            # Add console handler to logger
            self.logger.addHandler(self._console_handler)

    def set_file_logger(self, log_level, log_file_path):
        """ Set's the logging file path

        Args:
            log_file_path (str): valid path, including file type to store the
                logfile in. E.g: "folder_structure/log_file.txt"
            log_level (): file loglevel as specified in:
                https://docs.python.org/3/library/logging.html#logging-levels

        """
        self._log_file_path = log_file_path
        self._file_log_level = self._log_level_synonyms.get_flag(log_level)

        # Remove the old handler if there is one
        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)

        if self._file_log_level == 100:
            pass # deactivated logger
        else:
            # file log output format
            fh_formatter = logging.Formatter(
                '%(asctime)s %(name)s [%(threadName)-12s] [%(levelname)-7s] %(message)s')

            # Create new file handler
            self._file_handler = logging.FileHandler(self._log_file_path)
            self._file_handler.setLevel(self._file_log_level)
            self._file_handler.setFormatter(fh_formatter)

            # Add file handler to logger
            self.logger.addHandler(self._file_handler)

    def _create_logger(self, logger_name='esn_logger'):
        """ Creates and/or adjusts self.logger and sets it up for usage

        Args:
            logger_name (str): Name of the logger to be created. Used if one
                wants muliple different loggers with different properties (e.g.
                loglevels, etc

        Returns:
            None

        """
        self._logger_name = logger_name

        # Create logger. Note that the 2nd line is necessary, even if the
        # desired loglevel is not DEBUG
        self.logger = logging.getLogger(self._logger_name)
        self.logger.setLevel(logging.DEBUG)

    # def _update_logger(self):
    #     self.logger.addHandler(self._console_handler)



class SynonymDict():
    """ Custom dictionary wrapper to match synonyms with integer flags

    Internally the corresponding integer flags are used, but they are very much
    not descriptive so with this class one can define (str) synonyms for these
    flags, similar to how matplotlib does it

    Idea:
        self._synonym_dict = {flag1 : list of synonyms of flag1,
                              flag2 : list of synonyms of flag2,
                              ...}
    """

    def __init__(self):
        self._synonym_dict = {}
        # super().__init__()

    def add_synonyms(self, flag, synonyms):
        """ Assigns one or more synonyms to the corresponding flag

        Args:
            flag (int): flag to pair with the synonym(s)
            synonyms (): Synonym or iterable of synonyms. Technically any type
                is possible for a synonym but strings are highly recommended

        """

        # self.logger.debug("Add synonym(s) %s to flag %d"%(str(synonyms), flag))

        # Convert the synonym(s) to a list of synonyms
        if type(synonyms) is str:
            synonym_list = [synonyms]
        else:
            try:
                synonym_list = list(iter(synonyms))
            except TypeError:
                synonym_list = [synonyms]

        # make sure that the synonyms are not already paired to different flags
        for synonym in synonym_list:
            found_flag = self._find_flag(synonym)
            if flag == found_flag:
                # self.logger.info("Synonym %s was already paired to flag"
                #                  "%d" % (str(synonym), flag))
                synonym_list.remove(synonym)
            elif found_flag is not None:
                raise Exception("Tried to add Synonym %s to flag %d but"
                                " it was already paired to flag %d" %
                                (str(synonym), flag, found_flag))

        # add the synonyms
        if flag not in self._synonym_dict:
            self._synonym_dict[flag] = []
        self._synonym_dict[flag].extend(synonym_list)

    def _find_flag(self, synonym):
        """ Finds the corresponding flag to a given synonym.

        A flag is always also a synonym for itself

        Args:
            synonym ():

        Returns:
            flag (int_or_None): int if found, None if not

        """

        # self.logger.debug("Find flag for synonym %s"%str(synonym))

        flag = None
        if synonym in self._synonym_dict:
            flag = synonym
        else:
            for item in self._synonym_dict.items():
                if synonym in item[1]:
                    flag = item[0]

        return flag

    def get_flag(self, synonym):
        """ Finds the corresponding flag to a given synonym. Raises exception if
            not found

        see :func:`~SynonymDict._find_flag_from_synonym`

        """
        flag = self._find_flag(synonym)
        if flag is None:
            raise Exception("Flag corresponding to synonym %s not found" %
                            str(synonym))

        return flag

    #TODO: Add to tests:
    #
    # act_fct_flag_synonyms = rescomp.utilities.SynonymDict()
    # act_fct_flag_synonyms.add_synonyms(0, ["tanh simple", "simple"])
    # act_fct_flag_synonyms.add_synonyms(1, "tanh bias")
    # 0 == act_fct_flag_synonyms.get_flag(0)
    # 0 == act_fct_flag_synonyms.get_flag("tanh simple")
    # 1 == act_fct_flag_synonyms.get_flag("tanh bias")



def unique_key_by_value(dictionary, value):
    """ Finds key by value in a dict, raise exception if key is not unique

    Args:
        dictionary ():
        value ():

    Returns:
        key (): unique key corresponding to value

    """
    list_of_keys = keys_by_value(dictionary, value)
    if len(list_of_keys) == 1:
        key = list_of_keys[0]
    else:
        raise Exception("Key is NOT unique for the given value!\n"
                        "value=%s\nkeys found=%s," % (value, list_of_keys))

    return key


def keys_by_value(dictionary, value):
    """ Finds all keys corresponding to the given value in a dictionary

    Args:
        dictionary ():
        value ():

    Returns:
        list_of_keys (list): list of keys corresponding to value
    """
    list_of_keys = []
    list_of_items = dictionary.items()
    for item in list_of_items:
        if item[1] == value:
            list_of_keys.append(item[0])
    return list_of_keys


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
    with T time_steps use shape (T,d).

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

        #            check for dimension and time_steps
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

        starting_point = simulations.simulate_trajectory(
            sys_flag=reservoir.sys_flag,
            dt=reservoir.dt,
            time_steps=length,
            starting_point=original_start,
            print_switch=print_switch)[random_index]

        vals = simulations.simulate_trajectory(sys_flag=reservoir.sys_flag,
                                               dt=reservoir.dt,
                                               time_steps=timesteps,
                                               starting_point=starting_point,
                                               print_switch=print_switch)

    elif mode == 'fix_start':
        if starting_point is None:
            raise Exception('set starting_point to use fix_start')
        else:
            vals = simulations.simulate_trajectory(sys_flag=reservoir.sys_flag,
                                                   dt=reservoir.dt,
                                                   time_steps=timesteps,
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
