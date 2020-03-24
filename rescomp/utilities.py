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
import pandas
# import scipy.sparse
# import scipy.sparse.linalg
# import matplotlib.pyplot as plt
# import networkx as nx
from . import simulations
from ._version import __version__


class _ESNLogging:
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

        self._log_level_synonyms = _SynonymDict()
        self._log_level_synonyms.add_synonyms(0, ["NOTSET", "notset"])
        self._log_level_synonyms.add_synonyms(10, ["DEBUG", "debug"])
        self._log_level_synonyms.add_synonyms(20, ["INFO", "info"])
        self._log_level_synonyms.add_synonyms(30, ["WARNING", "warning"])
        self._log_level_synonyms.add_synonyms(40, ["ERROR", "error"])
        self._log_level_synonyms.add_synonyms(50, ["CRITICAL", "critical"])
        self._log_level_synonyms.add_synonyms(100, ["OFF", "off"])

        self._create_logger()
        self.set_console_logger(log_level="warning")
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
            self._console_handler = None   # deactivated handler
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
            log_level (): file loglevel as specified in:
                https://docs.python.org/3/library/logging.html#logging-levels
            log_file_path (str): path, including file type, to store the
                logfile in. E.g: "folder_structure/log_file.txt"

        """
        self._log_file_path = log_file_path
        self._file_log_level = self._log_level_synonyms.get_flag(log_level)

        # Remove the old handler if there is one
        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)

        if self._file_log_level == 100:
            self._file_handler = None   # deactivated handler
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


class _SynonymDict:
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


def read_pickle(path, compression="infer"):
    """ Load pickled (esn) object from file.

    Uses pandas functions internally.


    Args:
        path (str): File path where the pickled object will be loaded.
        compression ({'infer', 'gzip', 'bz2', 'zip', 'xz', None}) :
            default 'infer'
            For on-the-fly decompression of on-disk data. If 'infer', then use
            gzip, bz2, xz or zip if path ends in '.gz', '.bz2', '.xz',
            or '.zip' respectively, and no decompression otherwise.
            Set to None for no decompression.

    Returns:
        unpickled : same type as object stored in file

    """
    esn = pandas.read_pickle(path, compression)

    loaded_version = esn.get_internal_version()

    if __version__ != loaded_version:
        esn.logger.warning(
            "The rescomp package version used to create the loaded object is "
            "%s, while the package currently installed on the system is "
            "version %s)"%(loaded_version, __version__))

    return esn


def _unique_key_by_value(dictionary, value):
    """ Finds key by value in a dict, raise exception if key is not unique

    Args:
        dictionary ():
        value ():

    Returns:
        key (): unique key corresponding to value

    """
    list_of_keys = _keys_by_value(dictionary, value)
    if len(list_of_keys) == 1:
        key = list_of_keys[0]
    else:
        raise Exception("Key is NOT unique for the given value!\n"
                        "value=%s\nkeys found=%s," % (value, list_of_keys))

    return key


def _keys_by_value(dictionary, value):
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


def train_and_predict_input_setup(data, disc_steps=0, train_sync_steps=0, train_steps=None, pred_steps=None):
    """ Splits ESN input data for consecutive training and prediction

    This function is useful because there is an unintuitive overlap between
    x_train and x_pred of 1 time step which makes it easy to make mistakes

    Args:
        data (np.ndarray): data to be split/setup
        disc_steps (int): steps to discard completely before training begins
        train_sync_steps (int): steps to sync the reservoir with before training
        train_steps (int): steps to use for training and fitting w_in
        pred_steps (int): how many steps to predict the evolution for

    Returns:
        x_train (np.ndarray): input data for the training
        x_pred (np.ndarray): input data for the prediction

    """
    if train_steps is None: train_steps = data.shape[0] - disc_steps
    if pred_steps is None: pred_steps = data.shape[0] - train_steps - disc_steps

    x_train = data[disc_steps: disc_steps + train_sync_steps + train_steps]
    x_pred = data[disc_steps + train_sync_steps + train_steps - 1:
                  disc_steps + train_sync_steps + train_steps + pred_steps]

    return x_train, x_pred

