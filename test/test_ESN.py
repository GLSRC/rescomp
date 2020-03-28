# -*- coding: utf-8 -*-
""" Test if the resomp.esn module works as it should

"""

import unittest
import rescomp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
import time
import datetime
import pickle
from rescomp import simulations
import networkx as nx


def load_data_old(reservoir, data_input=None, mode='data_from_array', starting_point=None,
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


def save_realization_old(reservoir, filename='parameter/test_pickle_'):
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


def load_realization_old(reservoir, filename='parameter/test_pickle_', print_switch=False):
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


class ESNOld:
    """
    reservoir is a class for reservoir computing, using different network
    structures to predict (chaotic) time series

    :parameters:
    :sys: choose from lorenz.lorenz, lorenz.mod_lorenz and
        lorenz.roessler, eg. the system to predict
    :number_of_nodes: number of nodes for the actual reservoir network
    :input_dimension: dimension of the input to the reservoir
    :output_dimension: dimension of the output of the reservoir
    :type_of_network: network type, choose from 'random', 'scale_free' and
        'random_geometric' that is used as topological structure of the
        network
    :dt: time step for approximating the differential equation generating
        the timeseries
    :training_steps: number of datapoints used for training the output
        matrix
    :prediction_steps: number of prediction steps
    :discard_steps: number of steps, that are discarded befor recording r
        to synchronize the network with the input
    :regularization_parameter: weight for the Tikhonov-regularization term
        in the cost function (self.train())
    :spectral_radius: spectral radius of the actual reservoir network
    :edge_prob: the probability with which an edge is created in a random
        network
    :epsilon: threshold distance between y_test and y_pred for the
        prediction to be consided close (self.demerge_time())
    :extended_state: bool: if True: an extended state of the form [b,x,r]
        is used for updating the reservoir state, the prediction and
        fitting w_out.
        if False: only [r] is used
    :w_in_sparse: if True, each node only gets input from one dimension,
        if False, all dimensions are superimposed at each node
    :w_in_scale: Defines the absolute scale of uniformly distributed random
        numbers, centered at zero
    :activation_function_flag: selects the type of activation function
        (steepness, offset)
    :bias_scale: ###
    :normalize_data: boolean: if the time series should be normalized to zer
            mean and unit std.
    :r_squared: boolean: if the r vector should be squared for some nodes
        in each step
    """

    def __init__(self, sys_flag='mod_lorenz', network_dimension=500, input_dimension=3,
                 output_dimension=3, type_of_network='random',
                 dt=2e-2, training_steps=5000,
                 prediction_steps=5000, discard_steps=5000,
                 regularization_parameter=0.0001, spectral_radius=0.1,
                 avg_degree=6., epsilon=None, w_in_sparse=True,
                 w_in_scale=1., activation_function_flag='tanh', bias_scale=0.,
                 normalize_data=False, r_squared=False):

        if epsilon is None: epsilon = np.array([5, 10, 5])

        self.sys_flag = sys_flag
        self.n_dim = network_dimension
        self.type = type_of_network
        self.x_dim = input_dimension
        self.y_dim = output_dimension
        self.dt = dt
        self.training_steps = training_steps
        self.prediction_steps = prediction_steps
        self.discard_steps = discard_steps
        self.reg_param = regularization_parameter
        self.spectral_radius = spectral_radius
        self.avg_degree = float(avg_degree)  # = self.network.sum()/self.n_dim
        self.edge_prob = self.avg_degree / (self.n_dim - 1)
        self.b_out = np.ones((self.training_steps, 1))  # bias in fitting w_out
        self.epsilon = epsilon
        self.w_in_sparse = w_in_sparse
        self.w_in_scale = w_in_scale
        self.activation_function = None
        self.set_activation_function(activation_function_flag)
        self.bias_scale = bias_scale
        self.normalize_data = normalize_data
        self.r_squared = r_squared

        # topology of the network, adjacency matrix with entries 0. or 1.:
        self.binary_network = None  # calc_bina_scry_network() assigns values

        # load_data() assigns values to:
        self.x_train = None
        self.x_discard = None
        self.y_train = None
        self.y_test = None
        self.w_in = None

        # train() assigns values to:
        self.w_out = None
        self.r = None
        self.r2 = None #if self.r_squared is True

        # predict() assigns values to:
        self.r_pred = None
        self.r_pred2 = None #if self.squared is True
        self.y_pred = None
        self.noise = None

        # methods for analyizing assing values to:
        self.complete_network = None  # complete_network() assigns value

        self.timestamp = None  # timestamp; value assigned in save_realization()

        network_creation_attempts = 10
        for i in range(network_creation_attempts):
            try:
                self.create_network()
                self.vary_network() # previously: self.scale_network() can someone explain why? vary contains scale
            except ArpackNoConvergence:
                continue
            break
        else:
            raise Exception("Network creation during ESN init failed %d times"
                            %network_creation_attempts)

        self.set_bias()

        self.create_w_in()

    def create_w_in(self):
        if self.w_in_sparse:
            # w_in such that one element in each row is non-zero (Lu,Hunt, Ott 2018):
            self.w_in = np.zeros((self.n_dim, self.x_dim))
            for i in range(self.n_dim):
                random_x_coord = np.random.choice(np.arange(self.x_dim))
                self.w_in[i, random_x_coord] = np.random.uniform(
                    low=-self.w_in_scale, high=self.w_in_scale)  # maps input values to reservoir
        else:
            self.w_in = np.random.uniform(low=-self.w_in_scale,
                                          high=self.w_in_scale,
                                          size=(self.n_dim, self.x_dim))


    def create_network(self):
        # network type flags are handled:
        if self.type == 'random':
            network = nx.fast_gnp_random_graph(self.n_dim, self.edge_prob, seed=np.random)
        elif self.type == 'scale_free':
            network = nx.barabasi_albert_graph(self.n_dim, int(self.avg_degree / 2), seed=np.random)
            # radius, dimension have to be specified
        elif self.type == 'small_world':
            network = nx.watts_strogatz_graph(self.n_dim, k=int(self.avg_degree), p=0.1, seed=np.random)
        else:
            raise Exception("wrong self.type")

        # make a numpy array out of the network's adjacency matrix,
        # will be converted to scipy sparse in train(), predict()

        self.network = np.asarray(nx.to_numpy_matrix(network))

        self.calc_binary_network()

    def set_bias(self):
        #bias for each node to enrich the used interval of activation function:
        #if unwanted set self.bias_scale to zero.
        if self.bias_scale != 0:
            self.bias = self.bias_scale * np.random.uniform(low=-1.0, high=1.0, size=self.n_dim)
        else:
            self.bias = 0

    def set_activation_function(self, activation_function_flag):
        """
        method to change the activation function according to
        activation_function_flag variable
        """
        if activation_function_flag == 'tanh':
            self.activation_function = self.tanh
        else:
            raise Exception('activation_function_flag: '
                            + str(activation_function_flag)
                            + ' does not exist')

    def tanh(self, x, r):
        """
        standard activation function tanh()
        """
        return np.tanh(self.w_in @ x + self.network @ r + self.bias)

    def calc_binary_network(self):
        """
        returns a binary version of self.network to self.binary_network.
        """
        self.binary_network = np.zeros((self.network.shape))
        self.binary_network[np.nonzero(self.network)] = 1.

    def vary_network(self, print_switch=False):
        """
        Varies the weights of self.network, while conserving the topology.
        The non-zero elements of the adjacency matrix are uniformly randomized,
        and the matrix is scaled (self.scale_network()) to self.spectral_radius.
        """
        t0 = time.time()

        # contains tuples of non-zero elements:
        arg_binary_network = np.argwhere(self.network)

        network_variation_attempts = 10
        for i in range(network_variation_attempts):
            try:
                # uniform entries from [-0.5, 0.5) at non-zero locations:
                rand_shape = self.network[self.network != 0.].shape
                self.network[
                    arg_binary_network[:, 0], arg_binary_network[:, 1]] = \
                    np.random.random(size=rand_shape) - 0.5

                self.scale_network()

            except ArpackNoConvergence:
                continue
            break
        else:
            raise Exception("Network variation failed %d times"
                            %network_variation_attempts)

        t1 = time.time()
        if print_switch:
            print('varied non-zero entries in self.network in ', t1 - t0, 's')

    def scale_network(self):
        """
        Scale self.network, according to desired self.spectral_radius.
        Converts network in scipy.sparse object internally.
        """
        """
        Can cause problems due to non converging of the eigenvalue evaluation
        """
        self.network = scipy.sparse.csr_matrix(self.network)
        try:
            eigenvals = scipy.sparse.linalg.eigs(self.network,
                                                 k=1,
                                                 v0=np.ones(self.n_dim),
                                                 maxiter=1e3*self.n_dim)[0]
        except ArpackNoConvergence:
            print('Eigenvalue in scale_network could not be calculated!')
            raise

        maximum = np.absolute(eigenvals).max()
        self.network = ((self.spectral_radius / maximum) * self.network)
        self.network = self.network.toarray()

    def load_data(self, data_input=None, mode='data_from_array', starting_point=None,
                  add_noise=False, std_noise=0., print_switch=False):
        """
        Method to load data from a file or function (depending on mode)
        for training and testing the network.
        If the file does not exist, it is created according to the parameters.
        self._w_in is initialized.

        :parameters:
        :mode:
        - 'start_from_attractor' uses lorenz.record_trajectory for
            generating a timeseries, by randomly picking starting points from a
            given trajectory.
        - 'fix_start' passes
            for t in np.arange(self.discard_steps):
                self.r[0] = np.tanh(self._w_in @ self.x_discard[t] +
                            self.network @ self.r[0] )
            starting_point to lorenz.record_trajectory
        - 'data_from_file' loads a timeseries from file without further
        checking for reasonability - use with care!

        :add_noise: boolean: if normal distributed noise should be added to the
            imported timeseries
        :std_noise: standard deviation of the applied noise
        """

        load_data_old(self, data_input=data_input, mode=mode, starting_point=starting_point,
                      add_noise=add_noise, std_noise=std_noise, print_switch=print_switch)

    def train(self, print_switch=False):
        """
        Fits self.w_out, which connects the reservoir states and the input to
        the desired output, using linear regression and Tikhonov
        regularization.
        The convention is as follows: self.y[t+1] = self.w_out*self.r[t]
        Discards self.discard_steps steps befor recording the internal states
        of the reservoir (self.r),
        to synchronize the network dynamics with the input.

        Requires load_data() first, to pass values to x_train, y_train, y_test
        -> extend to test!
        Internally converts network in scipy.sparse object
        """
        t0 = time.time()

        # sparse, necessary for speed up in training loop
        self.network = scipy.sparse.csr_matrix(self.network)

        # states of the reservoir:
        self.r = np.zeros((self.training_steps, self.n_dim))

        # reservoir is synchronized with trajectory during discard_steps:
        for t in np.arange(self.discard_steps):
            self.r[0] = self.activation_function(self.x_discard[t], self.r[0])

        """
        the following step was included when Youssef proposed a revision of the
        timing. His concern was due to a missmatch between r and y
        (maybe we train the system to replicate the input not the next step
        -> has to be clarified!)
        """
        self.r[0] = self.activation_function(self.x_train[0], self.r[0])
        # states are then used to fit the target y_train:
        for t in range(self.training_steps - 1):
            self.r[t + 1] = self.activation_function(self.x_train[t + 1], self.r[t])
            # vector equation with
            # self.n_dim entries

        if self.r_squared:
            self.r2 = np.hstack((self.r, self.r**2))
        else:
            self.r2 = self.r

        self.w_out = np.linalg.solve((
                self.r2.T @ self.r2 + self.reg_param * np.eye(self.r2.shape[1])),
            (self.r2.T @ (self.y_train))).T

        t1 = time.time()
        if print_switch:
            print('training done in ', t1 - t0, 's')

        # array (backtransform from sparse)
        self.network = self.network.toarray()

    def predict(self, print_switch=False, prediction_noise=False, noise_scale=0.1):
        """
        Uses the self.w_out to predict output, using the network as
        recurrent network, feeding back in the (noisy) output.
        Internally converts network in scipy.sparse object
        """
        t0 = time.time()

        # sparse, necessary for speed up in training loop
        self.network = scipy.sparse.csr_matrix(self.network)

        ### predicting, fixed P, and using output as input again
        self.r_pred = np.zeros((self.prediction_steps, self.n_dim))
        self.y_pred = np.zeros((self.prediction_steps, self.y_dim))
        ### add noise to reinserted input
        if prediction_noise:
            self.noise = np.random.normal(loc=0.0, scale=noise_scale,
                                          size=(self.prediction_steps, self.y_dim))
        else:
            self.noise = np.zeros((self.prediction_steps, self.y_dim))

        # TODO: This line would be super wrong if y_train and r were not zeros
        # TODO: everywhere!
        self.r_pred[0] = self.activation_function(self.y_train[-1], self.r[-1])

        # transition from training to prediction
        if self.r_squared:
            self.r_pred2 = np.hstack((self.r_pred, self.r_pred**2))
        else:
            self.r_pred2 = self.r_pred
        self.y_pred[0] = self.w_out @ self.r_pred2[0]

        # prediction:
        for t in range(self.prediction_steps - 1):
            # update r:
            self.r_pred[t + 1] = self.activation_function(
                self.y_pred[t] + self.noise[t], self.r_pred[t])
            if self.r_squared:
                self.r_pred2[t + 1][:self.n_dim] = self.r_pred[t + 1]
                self.r_pred2[t + 1][self.n_dim:] = self.r_pred[t + 1] ** 2
                # self.r_pred2 = np.hstack((self.r_pred, self.r_pred**2))
            else:
                self.r_pred2 = self.r_pred
            # update y:
            self.y_pred[t + 1] = self.w_out @ self.r_pred2[t + 1]

        # array (backtransform from sparse)
        self.network = self.network.toarray()

        t1 = time.time()
        if print_switch:
            print('predicting done in ', t1 - t0, 's')

    def save_realization(self, filename='parameter/test_pickle_'):
        """
        Saves the network parameters (extracted with self.__dict__ to a file,
        for later reuse.

        pickle protocol version 2
        """
        save_realization_old(self, filename=filename)

    def load_realization(self, filename='parameter/test_pickle_', print_switch=False):
        """
        loads __dict__ from a pickled file and overwrites self.__dict__
        Original data lost!
        """
        load_realization_old(self, filename=filename, print_switch=print_switch)


class test_ESN(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.esn = rescomp.ESN()
        self.esn.set_console_logger("off")

    def tearDown(self):
        del self.esn
        np.random.seed(None)

    # TODO: Tests should be much less broad than this, but I am lazy.
    #   e.g: For a test, there is no reason to use simulated data, random
    #   data would have less dependencies.
    #   Also, this test only works on my (Sebastians) Laptop, the exact floats
    #   are different on every other system and hence the test has to be
    #   rewritten anyway to e.g. use the old ESN class as comparison point.
    #   Because the prediction difference on different systems is surprisingly
    #   large though, one should keep this test (or one like it) lying around
    #   somewhere though, as it demonstrates the absolute limit for the
    #   prediction of chaotic systems, purely due to the minimal differences in
    #   CPU architecture
    def test_sim_train_pred_mod_lorenz(self):
        train_sync_steps = 3
        train_steps = 3
        pred_steps = 2
        simulation_time_steps = train_sync_steps + train_steps + pred_steps

        starting_point = np.array([-2, -5, -1])
        sim_data = rescomp.simulations.simulate_trajectory(
            sys_flag='mod_lorenz', dt=2e-2, time_steps=simulation_time_steps,
            starting_point=starting_point)

        x_train, x_pred = rescomp.utilities.train_and_predict_input_setup(
            sim_data, train_sync_steps=train_sync_steps,
            train_steps=train_steps, pred_steps=pred_steps)

        # x_train = sim_data[:n_train_tot]
        # x_pred = sim_data[n_train_tot - 1: n_train_tot + n_predict]

        self.esn.create_network()

        self.esn.train(x_train, sync_steps=train_sync_steps)

        y_pred, y_test = self.esn.predict(x_pred, sync_steps=0)

        y_pred_desired = np.array(
            [[-8.009798237563704, -17.172409021843052, 3.689528434131512],
             [-8.199848199155639, -17.558818321636746, 3.879321151928091]])
        y_test_desired = np.array(
            [[-8.940650755161531, -18.88532291381985, 5.155862501900478],
             [-11.09758116927574, -22.68566274692776, 8.756832582764844]])

        np.testing.assert_equal(y_pred, y_pred_desired)
        np.testing.assert_equal(y_test, y_test_desired)


class test_ESNWrapper(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.esn = rescomp.ESNWrapper()
        self.esn.set_console_logger("off")

    def tearDown(self):
        del self.esn
        np.random.seed(None)

    def test_train_and_predict(self):
        disc_steps = 3
        train_sync_steps = 2
        train_steps = 5
        pred_steps = 4
        total_time_steps = disc_steps + train_sync_steps + train_steps + \
                           pred_steps

        x_dim = 3
        data = np.random.random((total_time_steps, x_dim))

        x_train, x_pred = rescomp.utilities.train_and_predict_input_setup(
            data, disc_steps=disc_steps, train_sync_steps=train_sync_steps,
            train_steps=train_steps, pred_steps=pred_steps)

        np.random.seed(1)
        self.esn.create_network()

        self.esn.train(x_train, train_sync_steps)
        y_pred_desired, y_test_desired = self.esn.predict(x_pred, sync_steps=0)

        self.tearDown()
        self.setUp()

        np.random.seed(1)
        self.esn.create_network()

        y_pred, y_test = self.esn.train_and_predict(
            data, disc_steps=disc_steps, train_sync_steps=train_sync_steps,
            train_steps=train_steps, pred_steps=pred_steps)

        # np.testing.assert_equal(data, data2)

        # np.testing.assert_equal(y_test, y_test_desired)
        np.testing.assert_equal(y_pred, y_pred_desired)


if __name__ == "__main__":
    unittest.main(verbosity=2)
