# -*- coding: utf-8 -*-
""" Calculating the lorenz63-attractor and other chaotic systems using 4th order runge kutta method

@author: aumeier, baur and herteux
"""
# from importlib import reload
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
import networkx as nx
import time
import pickle
# import matplotlib.pyplot as plt
# import datetime
import pandas.io.pickle
from . import utilities
from ._version import __version__

# dictionary defining synonyms for the different methods to generalize the
# reservoir state r(t) to a nonlinear fit for _w_out

class _ESNCore(utilities.ESNLogging):
    """ The non-reducible core of ESN RC training and prediction

    While technically possible to be used on it's own, this is very much not
    recommended. Use the child class ESN instead.

    Args:
        **kwargs (): Everything passed to the init of utilities.ESNLogging
    """

    def __init__(self):

        super().__init__()

        self._w_in = None

        self._network = None

        self._w_out = None

        self._act_fct = None

        self._w_out_fit_flag_synonyms = utilities.SynonymDict()
        self._w_out_fit_flag_synonyms.add_synonyms(0, ["linear_r", "simple"])
        self._w_out_fit_flag_synonyms.add_synonyms(1, "linear_and_square_r")

        self._w_out_fit_flag = None

        self._last_r = None
        self._last_r_gen = None
        # self._last_r = np.zeros(self._network.shape[0])
        # self._last_r_gen = self._r_to_generalized_r(self._last_r)

        self._reg_param = None

    def synchronize(self, x, save_r=False):
        """ Synchronize the reservoir state with the input time series

        Args:
            x (np.ndarray): shape (T,d)
            save_r (bool): If true, saves and returns r

        Returns:
            2dim np.ndarray containing all r(t) states if save_r is True
            None else

        """
        self.logger.debug('Start syncing the reservoir state')

        if self._last_r is None:
            self._last_r = np.zeros(self._network.shape[0])

        if save_r:
            r = np.zeros((x.shape[0], self._network.shape[0]))
            r[0] = self._act_fct(x[0], self._last_r)
            for t in np.arange(x.shape[0] - 1):
                r[t+1] = self._act_fct(x[t + 1], r[t])
            self._last_r = r[-1]
            return r
        else:
            for t in np.arange(x.shape[0]):
                self._last_r = self._act_fct(x[t], self._last_r)
            return None

    def _r_to_generalized_r(self, r):
        # This needs to work for both 2d r of shape (T, d) as well as 1d _last_r
        # of shape (d)
        if self._w_out_fit_flag is 0:
            return r
        elif self._w_out_fit_flag is 1:
            return np.hstack((r, r ** 2))
        else:
            raise Exception("self._w_out_fit_flag incorrectly specified")

    def _fit_w_out(self, y_train, r):
        """ Fit the output matrix self._w_out after training

        self._w_out connects the reservoir states and the input to
        the desired output, using linear regression and Tikhonov
        regularization.
        Note: There is no need for a save_r parameter as r_gen needs to be
        calculated anyway.

        Args:
            y_train (np.ndarray): Desired prediction from the reservoir states
            r (np.ndarray): reservoir states
        Returns:
            r_gen (np.ndarray): generalized nonlinear reservoir states form
        """

        self.logger.debug('Fit _w_out according to method%s' %
                          str(self._w_out_fit_flag))

        r_gen = self._r_to_generalized_r(r)

        self._w_out = np.linalg.solve(
            r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
            r_gen.T @ y_train).T

        return r_gen

    def _train_synced(self, x_train, w_out_fit_flag="simple"):
        """ Train a synchronized reservoir

        """

        self._w_out_fit_flag = \
            self._w_out_fit_flag_synonyms.get_flag(w_out_fit_flag)

        self.logger.debug('Start training')

        # The last value of r can't be used for the training, see comment below
        r = self.synchronize(x_train[:-1], save_r=True)

        # NOTE: This is slightly different than the old ESN as y_train was as
        # long as x_train, but shifted by one time step. Hence to get the same
        # results as for the old ESN one has to specify an x_train one time step
        # longer than before. Nonetheless, it's still true that r[t] is
        # calculated from x[t] and used to calculate y[t] (all the same t)
        y_train = x_train[1:]

        r_gen = self._fit_w_out(y_train, r)

        return r, r_gen

    def _predict_step(self, x):
        """ Predict a single time step

        Assumes a synchronized reservoir.
        Changes self._last_r and self._last_r_gen to stay synchronized to the new
        system state y

        Args:
            x (np.ndarray): input for the d-dim. system, shape (d)

        Returns:
            y (np.ndarray): the next time step as predicted from last_x, _w_out and
            _last_r, shape (d)
        
        """

        self._last_r = self._act_fct(x, self._last_r)
        self._last_r_gen = self._r_to_generalized_r(self._last_r)

        y = self._w_out @ self._last_r_gen

        return y


# TODO: Uni-indent the return part docstring of predict() and
#  train_and_predict()
class ESN(_ESNCore):
    # save_r: bool
    # should be called ESN
    # logfile and loglevel.
    # save instances/parameters to file
    # prediction noise in predict
    """
    input: training data and (list of) x_pred_test data, sync_steps
    specified/created: network, _w_in, activation_function
    internal: trained ESNCore instance
    output: prediction and desired prediction

    Goal: written such that one can easily implement both normal, full RC as
        well as local RC with arbitrary neighborhoods by calling this class with
        the right training and prediction data
    """

    # def __init__(self, network_dimension=500, input_dimension=None,
    #              type_of_network='random', avg_degree=6., spectral_radius=0.1,
    #              regularization_parameter=1e-5, w_in_sparse=True, w_in_scale=1.,
    #              act_fct_flag='tanh', bias_scale=0., **kwargs):

    def __init__(self):

        super().__init__()

        # create_network() assigns values to:
        self._n_dim = None  # network_dimension
        self._n_rad = None  # network_spectral_radius
        self._n_avg_deg = None  # network_average_degree
        self._n_edge_prob = None
        self._n_type_flag = None  #  network_type
        self._network = self._network

        # _create_w_in() which is called from train() assigns values to:
        self._w_in_sparse = None
        self._w_in_scale = None
        self._w_in = self._w_in

        # set_activation_function assigns values to:
        self._bias_scale = None
        self._bias = None
        self._act_fct_flag = None
        self._act_fct = self._act_fct

        # train() assigns values to:
        self._x_dim = None  # Typically called d
        self._reg_param = self._reg_param
        self._w_out = self._w_out
        # if save_input is true, train() also assigns values to:
        self._x_train_sync = None  # data used to sync before training
        self._x_train = None  # data used for training to fit w_out
        # if save_r is true, train() also assigns values to:
        self._r_train = None
        self._r_train_gen = None

        # predict() assigns values to:
        self._y_pred = None
        # if save_input is true, predict() also assigns values to:
        self._x_pred_sync = None  # data used to sync before prediction
        self._y_test = None  # data used to compare the prediction to
        # if save_r is true, predict() also assigns values to:
        self._r_pred = None
        self._r_pred_gen = None

        # Dictionary defining synonyms for the different ways to choose the
        # activation function. Internally the corresponding integers are used
        self._act_fct_flag_synonyms = utilities.SynonymDict()
        self._act_fct_flag_synonyms.add_synonyms(0, ["tanh_simple", "simple"])
        self._act_fct_flag_synonyms.add_synonyms(1, "tanh_bias")

        # Dictionary defining synonyms for the different ways to create the
        # network. Internally the corresponding integers are used
        self._n_type_flag_synonyms = utilities.SynonymDict()
        self._n_type_flag_synonyms.add_synonyms(0, ["random", "erdos_renyi"])
        self._n_type_flag_synonyms.add_synonyms(1, ["scale_free", "barabasi_albert"])
        self._n_type_flag_synonyms.add_synonyms(2, ["small_world", "watts_strogatz"])

        # Set during class creation, used during loading from pickle
        self._rescomp_version = __version__

        self.logger.debug("Create ESN instance")


    def _create_w_in(self):
        """ Create the input matrix w_in """
        self.logger.debug("Create w_in")

        if self._w_in_sparse:
            self._w_in = np.zeros((self._n_dim, self._x_dim))
            for i in range(self._n_dim):
                random_x_coord = np.random.choice(np.arange(self._x_dim))
                self._w_in[i, random_x_coord] = np.random.uniform(
                    low=-self._w_in_scale,
                    high=self._w_in_scale)  # maps input values to reservoir
        else:
            self._w_in = np.random.uniform(low=-self._w_in_scale,
                                          high=self._w_in_scale,
                                          size=(self._n_dim, self._x_dim))

    def create_network(self, n_dim=500, n_rad=0.1, n_avg_deg=6.0,
                       n_type_flag="erdos_renyi", network_creation_attempts=10):

        self.logger.debug("Create network")

        self._n_dim = n_dim
        self._n_rad = n_rad
        self._n_avg_deg = n_avg_deg
        self._n_edge_prob = self._n_avg_deg / (self._n_dim - 1)
        self._n_type_flag = self._n_type_flag_synonyms.get_flag(n_type_flag)

        for i in range(network_creation_attempts):
            try:
                self._create_network_connections()
                self._vary_network()
            except ArpackNoConvergence:
                continue
            break
        else:
            raise Exception("Network creation during ESN init failed %d times"
                            %network_creation_attempts)

    def _create_network_connections(self):

        if self._n_type_flag == 0:
            network = nx.fast_gnp_random_graph(self._n_dim, self._n_edge_prob,
                                               seed=np.random)
        elif self._n_type_flag == 1:
            network = nx.barabasi_albert_graph(self._n_dim,
                                               int(self._n_avg_deg / 2),
                                               seed=np.random)
        elif self._n_type_flag == 2:
            network = nx.watts_strogatz_graph(self._n_dim,
                                              k=int(self._n_avg_deg), p=0.1,
                                              seed=np.random)
        else:
            raise Exception("the network type %s is not implemented" %
                            str(self._n_type_flag))

        self._network = nx.to_numpy_array(network)
        # meh = nx.to_numpy_matrix(network)
        # self._network = np.asarray(nx.to_numpy_matrix(network))

    def _vary_network(self, network_variation_attempts=10):
        """ Varies the weights of self.network, while conserving the topology.

        The non-zero elements of the adjacency matrix are uniformly randomized,
        and the matrix is scaled (self.scale_network()) to self.spectral_radius.
        """

        # contains tuples of non-zero elements:
        arg_binary_network = np.argwhere(self._network)

        for i in range(network_variation_attempts):
            try:
                # uniform entries from [-0.5, 0.5) at non-zero locations:
                rand_shape = self._network[self._network != 0.].shape
                self._network[
                    arg_binary_network[:, 0], arg_binary_network[:, 1]] = \
                    np.random.random(size=rand_shape) - 0.5

                self._scale_network()

            except ArpackNoConvergence:
                self.logger.error(
                    'Network Variaion failed! -> Try agin!')

                continue
            break
        else:
            #TODO: Better logging of exceptions
            self.logger.error("Network variation failed %d times"
                            % network_variation_attempts)
            raise ArpackNoConvergence

    def _scale_network(self):
        """ Scale self.network, according to desired self.spectral_radius.

        Converts network in scipy.sparse object internally.
        """
        """
        Can cause problems due to non converging of the eigenvalue evaluation
        """
        self._network = scipy.sparse.csr_matrix(self._network)
        try:
            eigenvals = scipy.sparse.linalg.eigs(
                self._network, k=1, v0=np.ones(self._n_dim),
                maxiter=1e3 * self._n_dim)[0]
        except ArpackNoConvergence:
            self.logger.error('Eigenvalue calculation in scale_network failed!')
            raise

        maximum = np.absolute(eigenvals).max()
        self._network = ((self._n_rad / maximum) * self._network)

    def set_network(self, network):
        """ Set the network by passing a matrix or a file path

        Calculates the corresponding properties to match the new network

        Args:
            network (nd_array_or_csr_matrix_str):

        Returns:

        """
        raise Exception("Not yet implemented")

    def _set_activation_function(self, act_fct_flag, bias_scale=0):
        """ Set the activation function to the one corresponding to act_fct_flag

        Args:
            act_fct_flag (): flag corresponding to the activation function one
                wants to use
            bias_scale (float): Bias to be used in some activation functions
                (currently only in :func:`~ESN._act_fct_tanh_bias`)

        """
        self.logger.debug("Set activation function to flag: %s" % act_fct_flag)

        # self._act_fct_flag = act_fct_flag
        self._act_fct_flag = self._act_fct_flag_synonyms.get_flag(act_fct_flag)

        self._bias_scale = bias_scale
        self._bias = self._bias_scale * np.random.uniform(low=-1.0, high=1.0,
                                                          size=self._n_dim)

        if self._act_fct_flag == 0:
            self._act_fct = self._act_fct_tanh_simple
        elif self._act_fct_flag == 1:
            self._act_fct = self._act_fct_tanh_bias
        else:
            raise Exception('self._act_fct_flag %s does not have a activation '
                            'function implemented!' % str(self._act_fct_flag))

    def _act_fct_tanh_simple(self, x, r):
        """ Standard activation function of the elementwise np.tanh()

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            (np.ndarray) n-dim

        """

        return np.tanh(self._w_in @ x + self._network @ r)

    def _act_fct_tanh_bias(self, x, r):
        """ Activation function of the elementwise np.tanh() with added bias

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            (np.ndarray) n-dim

        """

        return np.tanh(self._w_in @ x + self._network @ r + self._bias)

    def train(self, x_train, sync_steps, reg_param=1e-5, w_in_scale=1.0,
                      w_in_sparse=True, act_fct_flag='tanh_simple', bias_scale=0,
                      save_r=False, save_input=False, **kwargs):
        """ Train the reservoir after synchronizing it

        Args:
            x_train (np.ndarray): Input data used to synchronize and then train
                the reservoir
            sync_steps (int): How many steps to use for synchronization before
                the prediction starts
            reg_param (float): weight for the Tikhonov-regularization term
            w_in_scale (float): maximum absolut value of the (random) w_in
                elements
            w_in_sparse (bool): If true, creates w_in such that one element in
                each row is non-zero (Lu,Hunt, Ott 2018)
            save_r (bool): If true, saves r(t) internally
            save_input (bool):If true, saves the input data internally

        """
        self._reg_param = reg_param
        self._w_in_scale = w_in_scale
        self._w_in_sparse = w_in_sparse
        self._x_dim = x_train.shape[1]
        self._create_w_in()

        self._set_activation_function(act_fct_flag=act_fct_flag,
                                      bias_scale=bias_scale)

        if sync_steps != 0:
            x_sync = x_train[:sync_steps]
            x_train = x_train[sync_steps:]
            self.synchronize(x_sync)
        else:
            x_sync = None

        if save_input:
            self._x_train_sync = x_sync
            self._x_train = x_train

        if save_r:
            self._r_train, self._r_train_gen = self._train_synced(x_train)
        else:
            self._train_synced(x_train)

    def predict(self, x_pred, sync_steps, pred_steps=None,
                save_r=False, save_input=False, **kwargs):
        """ Predict the system evolution after synchronizing the reservoir

        Changes self._last_r and self._last_r_gen to stay synchronized to the new
        system state

        Args:
            x_pred (np.ndarray): Input data used to synchronize the reservoir,
                and then use the rest to predict
            sync_steps (int): How many steps to use for synchronization before
                the prediction starts
            pred_steps (int): How many steps to predict
            save_r (bool): If true, saves r(t) internally
            save_input (bool):If true, saves the input data internally

        Returns:
            (tuple): tuple containing:
                y_pred (np.ndarray): Predicted future states
                    y_test (np.ndarray_or_None): Data taken from the input to
                    compare the prediction with. If the prediction were
                    "perfect" y_pred and y_test would be equal. Be careful
                    though, y_test might be shorter than y_pred, or even None,
                    if pred_steps is not None

        """

        if pred_steps is None:
            pred_steps = x_pred.shape[0] - sync_steps - 1
            
        if len(x_pred)>sync_steps+pred_steps+1:
            x_pred=x_pred[:sync_steps+pred_steps+1]

        # Automatically generates a y_test to compare the prediction against, if
        # the input data is longer than the number of synchronization tests
        if sync_steps == 0:
            x_sync = None
            y_test = x_pred[1:]
        elif sync_steps <= x_pred.shape[0]:
            x_sync = x_pred[:sync_steps]
            y_test = x_pred[sync_steps + 1:]
        else:
            x_sync = x_pred[:-1]
            y_test = None

        if save_input:
            self._x_pred_sync = x_sync
            self._y_test = y_test

        if x_sync is not None:
            self.synchronize(x_sync)

        self.logger.debug('Start Prediction')

        self._y_pred = np.zeros((pred_steps, x_pred.shape[1]))

        self._y_pred[0] = self._predict_step(x_pred[sync_steps])

        if save_r:
            self._r_pred = np.zeros((pred_steps, self._network.shape[0]))
            self._r_pred_gen = self._r_to_generalized_r(self._r_pred)
            self._r_pred[0] = self._last_r
            self._r_pred_gen[0] = self._last_r_gen

            for t in range(pred_steps - 1):
                self._y_pred[t + 1] = self._predict_step(self._y_pred[t])

                self._r_pred[t + 1] = self._last_r
                self._r_pred_gen[t + 1] = self._last_r_gen

        else:
            for t in range(pred_steps - 1):
                self._y_pred[t + 1] = self._predict_step(self._y_pred[t])

        return self._y_pred, y_test

    def get_network(self, ):
        raise Exception("Not yet implemented")

    def get_w_in(self, ):
        raise Exception("Not yet implemented")

    def get_activation_function(self, ):
        raise Exception("Not yet implemented")

    def get_training(self, ):
        raise Exception("Not yet implemented")

    def get_prediction(self, ):
        raise Exception("Not yet implemented")

    def get_internal_version(self):
        """ Returns the rescomp package version used to create the class instance

        Returns:
            str: Rescomp package version

        """
        return self._rescomp_version

    def to_pickle(self, path, compression="infer",
                  protocol=pickle.HIGHEST_PROTOCOL):
        """ Pickle (serialize) object to file.

        Disables logging as logging handlers can not be pickled.
        Uses pandas functions internally.

        Args:
            path (str): File path where the pickled object will be stored.
            compression ({'infer', 'gzip', 'bz2', 'zip', 'xz', None}):
                default 'infer'
                A string representing the compression to use in the output file. By
                default, infers from the file extension in specified path.
            protocol (int): Int which indicates which protocol should be used by the pickler,
            default HIGHEST_PROTOCOL (see [1]_ paragraph 12.1.2). The possible
            values are 0, 1, 2, 3, 4. A negative value for the protocol
            parameter is equivalent to setting its value to HIGHEST_PROTOCOL.

            .. [1] https://docs.python.org/3/library/pickle.html
        """

        self.logger.debug("Save to file %s, turn off internal logger to do so"%path)

        self.set_console_logger("off")
        self.set_file_logger("off", None)

        pandas.io.pickle.to_pickle(self, path, compression=compression,
                                   protocol=protocol)


class ESNWrapper(ESN):
    """ Convenience functions for the ESN class
    """

    def __init__(self):
        super().__init__()
        self.logger.debug("Create ESNWrapper instance")

    def train_and_predict(self, x_data, train_sync_steps, train_steps,
                          pred_steps=None, disc_steps=0, **kwargs):
        """ Train, then predict the evolution directly following the train data

        Args:
            x_data (np.ndarray): Data used for synchronization, training and
                prediction (start and comparison)
            train_sync_steps (int): Steps to synchronize the reservoir with
                before the 'real' training begins
            train_steps (int): Steps to use for training and fitting w_in
            pred_steps (int): How many steps to predict the evolution for
            **kwargs (): further arguments passed to train() and predict()

        Returns:
            y_pred (np.ndarray): Predicted future states
                y_test (np.ndarray_or_None): Data taken from the input to compare
                the prediction with. If the prediction were "perfect" y_pred and
                y_test would be equal. Be careful though, y_test might be
                shorter than y_pred, or even None, if pred_steps is not None

        """
        x_train, x_pred = utilities.train_and_predict_input_setup(
            x_data, disc_steps=disc_steps, train_sync_steps=train_sync_steps,
            train_steps=train_steps, pred_steps=pred_steps)

        self.train(x_train, train_sync_steps, **kwargs)

        y_pred, y_test = self.predict(x_pred, sync_steps=0,
                        pred_steps=pred_steps, **kwargs)

        return y_pred, y_test


    def predict_multiple(self, ):
        """ Predict system evolution from multiple different starting conditions
        """
        raise Exception("Not yet implemented")

    def calc_binary_network(self):
        """ Returns a binary version of self._network
        """
        raise Exception("Not yet implemented")

    def remove_nodes(self, split):
        """ See the out-commented remove_nodes fct in measures
        """
        raise Exception("Not yet implemented")


