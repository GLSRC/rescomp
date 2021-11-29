# -*- coding: utf-8 -*-
""" Various utility functions for RC and the ESN classes generally """

import numpy as np
import logging
import sys
import pandas
import inspect
import fractions
import pkg_resources
from ._version import __version__

_rescomp_loggers = {}
_rescomp_logger_counter = 0


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

        if self._console_log_level == 100:
            self._console_handler = None  # deactivated handler
        else:
            # console log output format
            ch_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)-7s] %(message)s',
                datefmt='%m-%d %H:%M:%S')

            # Create new console handler
            self._console_handler = logging.StreamHandler(stream=sys.stdout)
            self._console_handler.setLevel(self._console_log_level)
            self._console_handler.setFormatter(ch_formatter)

            # Remove the old handler if there is one
            if self._console_handler in self.logger.handlers:
                self.logger.removeHandler(self._console_handler)

            # Add console handler to logger
            self.logger.addHandler(self._console_handler)

        # for h in self.logger.handlers:
        #     print('     %s' % h)
        #
        #     for nm, lgr in logging.Logger.manager.loggerDict.items():
        #         print('+ [%-20s] %s ' % (nm, lgr))
        #         if not isinstance(lgr, logging.PlaceHolder):
        #             for h in lgr.handlers:
        #                 print('     %s' % h)

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

        global _rescomp_logger_counter
        self._logger_name = logger_name + str(_rescomp_logger_counter)

        global _rescomp_loggers

        if self._logger_name in _rescomp_loggers.keys():
            self.logger = _rescomp_loggers[self._logger_name]
            self.logger.propagate = False
        else:

            # Create logger
            self.logger = logging.getLogger(self._logger_name)

            # for h in self.logger.handlers:
            #     print('     %s' % h)

            # self.logger.handlers.clear()

            # Hack to avoid multiple logging outputs resulting from calling
            # logging.getLogger mutliple times as e.g. happens when you create
            # multiple instances of this class
            # Delete old handlers attached to previous loggers with the same name
            # Happens e.g. when creating a class instance multiple times
            # TODO: Do this correctly!
            # if self.logger.hasHandlers():
            #     self.logger.removeHandler(self.logger.handlers[0])
            #
            self.logger.propagate = False

            # Note that this line is necessary, even if the desired loglevel
            # is not DEBUG
            self.logger.setLevel(logging.DEBUG)

            _rescomp_loggers[self._logger_name] = self.logger
            # _rescomp_loggers["a"] = "b"
            _rescomp_logger_counter += 1

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
            synonym (): Thing to find the synonym for

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

    # TODO: Add to tests:
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

    loaded_version = esn.get_instance_version()

    if __version__ != loaded_version:
        esn.logger.warning(
            "The rescomp package version used to create the loaded object is "
            "%s, while the package currently installed on the system is "
            "version %s)" % (loaded_version, __version__))

    return esn


def _unique_key_by_value(dictionary, value):
    """ Finds key by value in a dict, raise exception if key is not unique

    Args:
        dictionary ():
        value ():

    Returns:
        (): unique key corresponding to value

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
        list: list_of_keys, list of keys corresponding to value
    """
    list_of_keys = []
    list_of_items = dictionary.items()
    for item in list_of_items:
        if item[1] == value:
            list_of_keys.append(item[0])
    return list_of_keys


def _remove_invalid_args(func, args_dict):
    """Return dictionary of valid args and kwargs with invalid ones removed

    Adjusted from:
    https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives

    Args:
        func (fct): function to check if the arguments are valid or not
        args_dict (dict): dictionary of arguments

    Returns:
        dict: dictionary of valid arguments

    """
    valid_args = inspect.signature(func).parameters
    # valid_args = func.func_code.co_varnames[:func.func_code.co_argcount]
    return dict((key, value) for key, value in args_dict.items() if key in valid_args)


def train_and_predict_input_setup(data, disc_steps=0, train_sync_steps=0,
                                  train_steps=None, pred_sync_steps=0,
                                  pred_steps=None):
    """ Splits ESN input data for consecutive training and prediction

    This function is useful because there is an unintuitive overlap between
    x_train and x_pred of 1 time step which makes it easy to make mistakes

    Args:
        data (np.ndarray): data to be split/setup
        disc_steps (int): steps to discard completely before training begins
        train_sync_steps (int): steps to sync the reservoir with before training
        train_steps (int): steps to use for training and fitting w_in
        pred_sync_steps (int): steps to sync the reservoir with before prediction
        pred_steps (int): how many steps to predict the evolution for

    Returns:
        tuple: 2-element tuple containing:

        - **x_train** (*np.ndarray*): input data for the training
        - **x_pred** (*np.ndarray*): input data for the prediction

    """
    if train_steps is None:
        x_train = data[disc_steps:]
    else:
        x_train = data[disc_steps: disc_steps + train_sync_steps + train_steps]

    if pred_steps is None:
        x_pred = data[disc_steps + train_sync_steps + train_steps - 1:]
    else:
        x_pred = data[disc_steps + train_sync_steps + train_steps - 1:
                      disc_steps + train_sync_steps + train_steps +
                      pred_sync_steps + pred_steps]

    return x_train, x_pred


def _find_nth_substring(haystack, substring, n):
    """ Finds the position of the n-th occurrence of a substring

    Args:
        haystack (str): Main string to find the occurrences in
        substring (str): Substring to search for in the haystack string
        n (int): The occurrence number of the substring to be located. n > 0

    Returns:
        int_or_None: Position index of the n-th substring occurrence in the
        haystack string if found. None if not found.

    """
    parts = haystack.split(substring, n)

    if n <= 0 or len(parts) <= n:
        return None
    else:
        return len(haystack) - len(parts[-1]) - len(substring)


def _get_internal_version():
    """ Returns the internal rescomp version as specified in rescomp._version

    Returns:
        str: int_version, internal rescomp package version

    """
    int_version = __version__
    return int_version


def _get_environment_version():
    """ Returns the rescomp version as specified in the python environment

    Returns:
        str: env_version, environment rescomp package version

    """
    try:
        import rescomp
        env_version = pkg_resources.require("rescomp")[0].version
    except (ImportError, pkg_resources.DistributionNotFound):
        env_version = "0.0.0"
    return env_version


def _compare_version_file_vs_env(segment_threshold="minor"):
    """ Compare version file with the version number in the python environment

    Compares the internally defined version number of the rescomp package
    as specified in rescomp._version with the version number specified in the
    activate python environment up to the defined component threshold

    Args:
        component_threshold (str): Defines up to which segment of the version
            string the versions are compared. Possible flags are:

                - "major": major version numbers are compared
                - "minor": major and minor version numbers are compared
                - "micro": major, minor and micro version numbers are compared

    Returns:
        bool: True if internal and environemnt versions are the same up to and
            including the specified threshold. False if not.

    """
    int_version = _get_internal_version()
    env_version = _get_environment_version()

    if segment_threshold == "major":
        int_version = int_version[:_find_nth_substring(int_version, '.', 1)]
        env_version = env_version[:_find_nth_substring(env_version, '.', 1)]
    elif segment_threshold == "minor":
        int_version = int_version[:_find_nth_substring(int_version, '.', 2)]
        env_version = env_version[:_find_nth_substring(env_version, '.', 2)]
    elif segment_threshold == "micro":
        pass
    else:
        raise Exception("segment_threshold %s not recognized" % segment_threshold)

    return int_version == env_version


def _is_number(s):
    """ Tests if input is a number

    Args:
        s (any): Object you want to check

    Returns:
        True if s is a number, False if not
    """
    try:
        float(s)
        return True
    except ValueError:
        try:
            fractions.Fraction(s)
            return True
        except ValueError:
            return False
