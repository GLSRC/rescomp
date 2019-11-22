"""
:author: aumeier, edited slightly by baur
:date: 2019/04/26
:license: ???
"""

#from importlib import reload
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
#import matplotlib.pyplot as plt
import networkx as nx
import pickle
import time
import datetime

import sys
sys.path.append("..")
sys.path.append(".")
try: import lorenz_rescomp
except ModuleNotFoundError: from . import lorenz_rescomp

class res_core(object):
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
    :dt: timestep for approximating the differential equation generating
        the timeseries
    :training_steps: number of datapoints used for training the output
        matrix
    :prediction_steps: number of prediction steps
    :discard_steps: number of steps, that are discarded befor recording r
        to synchronize the network with the input        
    :regularization_parameter: weight for the Tikhonov-regularization term
        in the cost function (self.train())
    :spectral_radius: spectral radius of the actual reservoir network
    :input_weight: weight of the input state over the reservoir state in
        the neurons
    :edge_prob: the probability with which an edge is created in a random
        network
    :epsilon: threshold distance between y_test and y_pred for the
        prediction to be consided close (self.demerge_time())
    :extended_state: bool: if True: an extended state of the form [b,x,r]
        is used for updating the reservoir state, the prediction and
        fitting W_out.
        if False: only [r] is used
    :W_in_sparse: if True, each node only gets input from one dimension,
        if False, all dimensions are superimposed at each node
    :W_in_scale: Defines the absolute scale of uniformly distributed random
        numbers, centered at zero
    :activation_function_flag: selects the type of activation function 
        (steepness, offset).
    """

    def __init__(self, sys_flag='mod_lorenz', N=500, input_dimension=3,
                 output_dimension=3, type_of_network='random',
                 dt=2e-2, training_steps=5000,
                 prediction_steps=5000, discard_steps=5000,
                 regularization_parameter=0.0001, spectral_radius=0.1,
                 input_weight=1., avg_degree=6., epsilon=None,
                 extended_states=False, W_in_sparse=True, W_in_scale=1.,
                 activation_function_flag='tanh'):

        if epsilon == None: epsilon = np.array([5, 10, 5])

        self.sys_flag = sys_flag
        self.N = N
        self.type = type_of_network
        self.xdim = input_dimension
        self.ydim = output_dimension
        self.dt = dt
        self.training_steps = training_steps
        self.prediction_steps = prediction_steps
        self.discard_steps = discard_steps
        self.reg_param = regularization_parameter
        self.spectral_radius = spectral_radius
        self.input_weight = input_weight
        self.avg_degree = float(avg_degree)  # = self.network.sum()/self.N
        self.edge_prob = self.avg_degree / (self.N - 1)
        self.b_out = np.ones((self.training_steps, 1))  # bias in fitting W_out
        self.extended_states = extended_states
        self.epsilon = epsilon
        self.W_in_sparse = W_in_sparse
        self.W_in_scale = W_in_scale
        self.activation_function = None
        self.set_activation_function(activation_function_flag)

        self.base_class_test_variable = 17.
        # topology of the network, adjacency matrix with entries 0. or 1.:
        self.binary_network = None  # calc_bina_scry_network() assigns values

        # load_data() assigns values to:
        self.x_train = None
        self.x_discard = None
        self.y_train = None
        self.y_test = None
        self.W_in = None

        # train() assigns values to:
        self.W_out = None
        self.r = None

        # predict() assigns values to:
        self.r_pred = None
        self.y_pred = None
        self.noise = None

        # methods for analyizing assing values to:
        self.complete_network = None  # complete_network() assigns value

        self.timestamp = None  # timestamp; value assigned in save_realization()

        # network type flags are handled:
        if type_of_network == 'random':
            network = nx.fast_gnp_random_graph(self.N, self.edge_prob)

        elif type_of_network == 'scale_free':
            network = nx.barabasi_albert_graph(self.N, int(self.avg_degree / 2))
            # radius, dimension have to be specified
        elif type_of_network == 'small_world':
            network = nx.watts_strogatz_graph(self.N, k=int(self.avg_degree), p=0.1)
        else:
            raise Exception("wrong self.type_of_network")

        # make a numpy array out of the network's adjacency matrix,
        # will be converted to scipy sparse in train(), predict()
        self.network = np.asarray(nx.to_numpy_matrix(network))

        self.calc_binary_network()

        if self.W_in_sparse:
            # W_in such that one element in each row is non-zero (Lu,Hunt, Ott 2018):
            self.W_in = np.zeros((self.N, self.xdim))
            for i in range(self.N):
                random_x_coord = np.random.choice(np.arange(self.xdim))
                self.W_in[i, random_x_coord] = np.random.uniform(
                    low=-self.W_in_scale, high=self.W_in_scale)  # maps input values to reservoir
        else:
            self.W_in = np.random.uniform(low=-self.W_in_scale,
                                          high=self.W_in_scale,
                                          size=(self.N, self.xdim))

    #    def __str__(self):
    #        return str('measures.reservoir('+str(self.N)+')')
    #
    #    def __repr__(self):
    #        pass

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
        return np.tanh(self.input_weight * self.W_in @ x + self.network @ r)

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

        # uniform entries from [-0.5, 0.5) at the former non-zero locations:
        self.network[arg_binary_network[:, 0],
                     arg_binary_network[:, 1]] = np.random.random(
            size=self.network[self.network != 0.].shape) - 0.5

        self.scale_network()

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
            eigenvals = scipy.sparse.linalg.eigs(self.network, k=1, which='LM')[0]
            maximum = np.absolute(eigenvals).max()

            self.network = ((self.spectral_radius / maximum) * self.network)
        except:
            print('scaling failed due to non-convergence of eigenvalue \
            evaluation.')
        self.network = self.network.toarray()

    #        self.network = self.spectral_radius*(self.network/np.absolute(
    #            np.linalg.eigvals(self.network)).max())

    def load_data(self, mode='start_from_attractor', starting_point=None,
                  add_noise=False, std_noise=0., print_switch=False,
                  data_input=None):
        """
        Method to load data from a file or function (depending on mode)
        for training and testing the network.
        If the file does not exist, it is created according to the parameters.
        self.W_in is initialized.
        
        :parameters:
        :mode:
        - 'start_from_attractor' uses lorenz.record_trajectory for 
            generating a timeseries, by randomly picking starting points from a 
            given trajectory.
        - 'fix_start' passes        for t in np.arange(self.discard_steps):
            self.r[0] = np.tanh(
                self.input_weight *
                np.matmul(self.W_in, self.x_discard[t]) + \
                np.matmul(self.network, self.r[0]) ) starting_point to lorenz.record_trajectory
        - 'data_from_file' loads a timeseries from file without further
        checking for reasonability - use with care!
        
        :add_noise: boolean: if normal distributed noise should be added to the
            imported timeseries
        :std_noise: standard deviation of the applied noise
        """
        t0 = time.time()

        # minimum size for vals, lorenz.save_trajectory has to evaluate:
        timesteps = 1 + self.discard_steps + self.training_steps \
                    + self.prediction_steps

        if print_switch:
            print('mode: ', mode)

        if mode == 'data_from_file':

            #            check for dimension and timesteps
            #            self.discard_steps + self.training_steps + \
            #            self.prediction_steps (+1) == vals.shape

            vals = data_input

            vals -= vals.meactivation_funcan(axis=0)
            vals *= 1 / vals.std(axis=0)
            print(vals.shape)

        elif mode == 'start_from_attractor':
            length = 50000
            original_start = np.array([-2.00384153, -5.34877257, -1.20401106])
            random_index = np.random.choice(np.arange(10000, length, 1))
            if print_switch:
                print('random index for starting_point: ', random_index)

            starting_point = lorenz_rescomp.record_trajectory(
                sys_flag=self.sys_flag,
                dt=self.dt,
                timesteps=length,
                starting_point=original_start)[random_index]

            vals = lorenz_rescomp.record_trajectory(sys_flag=self.sys_flag,
                    dt=self.dt, timesteps=timesteps, starting_point=starting_point)

        elif mode == 'fix_start':
            if starting_point is None:
                raise Exception('set starting_point to use fix_start')
            else:
                vals = lorenz_rescomp.record_trajectory(sys_flag=self.sys_flag,
                        dt=self.dt,  timesteps=timesteps, starting_point=starting_point)
        else:
            raise Exception(mode, ' mode not recognized')
            # print(mode, ' mode not recognized')
        # print('data loading successfull')
        if add_noise:
            vals += np.random.normal(scale=std_noise, size=vals.shape)
            print('added noise with std_dev: ' + str(std_noise))

        # define local variables for test/train split:
        n_test = self.prediction_steps
        n_train = self.training_steps + self.discard_steps
        n_discard = self.discard_steps

        # sketch of test/train split:
        # [--n_discard --|--n_train--|--n_test--]
        # and y_train is shifted by 1
        self.x_train = vals[n_discard:n_train, :]  # input values driving reservoir
        self.x_discard = vals[:n_discard, :]
        self.y_train = vals[n_discard + 1:n_train + 1, :]  # +1
        self.y_test = vals[n_train + 1:n_train + n_test + 1, :]  # +1

        # check, if y_test has prediction_steps:
        # should be extended to a real test_func!
        if self.y_test.shape[0] != self.prediction_steps:
            print('length(y_test) [' + str(self.y_test.shape[0])
                  + '] != prediction_steps [' + str(self.prediction_steps) + ']')

        t1 = time.time()
        if print_switch:
            print('input (x) and target (y) loaded in ', t1 - t0, 's')

    def train(self, print_switch=False):
        """
        Fits self.W_out, which connects the reservoir states and the input to
        the desired output, using linear regression and Tikhonov
        regularization.
        The convention is as follows: self.y[t+1] = self.W_out*self.r[t]
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
        self.r = np.zeros((self.training_steps, self.N))

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
            # self.N entries
        '''
        zimmermann, parlitz paper uses not only the reservoir states but also
        the current input for predicting an outcome
        the x(input) values are appended to the r(reservoir state) values
        transpose to match the formula from the paper
        W_out will be used as follows: 
        '''

        """
        timing for extended_states NOT adapted to new_timing yet        
       
        if self.extended_states:
            #create the extended state:[b_out, r, x] and time as 2nd dimension:
            #X = np.concatenate((self.b_out, self.r, self.x_train), axis=1).T
            #create the extended state:[r, x] and time as 2nd dimension:
            X = np.concatenate((self.r, self.x_train), axis=1).T
        
        else:
        """
        # X = self.r.T
        # Y = self.y_train.T

        # actual calculation of self.W_out:

        #        t_old = time.time()
        #        self.W_out_old = np.matmul(
        #            np.matmul(Y,X.T), np.linalg.inv(np.matmul(X,X.T)
        #            + self.reg_param*np.eye(X.shape[0])))
        #        print('old: ')
        #        print(time.time() - t_old)
        """
        this version should be matched with the old one, and then implemented
        ultimately.
        """
        self.W_out = np.linalg.solve((
                self.r.T @ self.r + self.reg_param * np.eye(self.r.shape[1])),
            (self.r.T @ (self.y_train))).T

        t1 = time.time()
        if print_switch:
            print('training done in ', t1 - t0, 's')

        # array (backtransform from sparse)
        self.network = self.network.toarray()

    def predict(self, print_switch=False, prediction_noise=False, noise_scale=0.1):
        """
        Uses the self.W_out to predict output, using the network as
        recurrent network, feeding back in the (noisy) output.
        Internally converts network in scipy.sparse object   
        """
        t0 = time.time()

        # sparse, necessary for speed up in training loop
        self.network = scipy.sparse.csr_matrix(self.network)

        ### predicting, fixed P, and using output as input again
        self.r_pred = np.zeros((self.prediction_steps, self.N))
        self.y_pred = np.zeros((self.prediction_steps, self.ydim))
        ### add noise to reinserted input
        if prediction_noise:
            self.noise = np.random.normal(loc=0.0, scale=noise_scale,
                                          size=(self.prediction_steps, self.ydim))
        else:
            self.noise = np.zeros((self.prediction_steps, self.ydim))

        if self.extended_states:
            print('Incorrect timing when using extenden_states!')
        #            """
        #            should I implement b as well?
        #            """
        #            #create the extended state:[b_out, r, x] and time as 2nd dimension:
        #            #X = np.concatenate((self.b_out, self.r, self.x_train), axis=1).T
        #
        #            #create the extended state:[r, x] and time as 2nd dimension:
        #
        #            self.y_pred[0] = np.matmul(self.W_out,
        #                                np.hstack((self.r[-1], self.x_train[-1])))
        #
        #            for t in range(self.prediction_steps - 1):
        #                """
        #                Add noise
        #                """
        #                #update r:
        #                self.r_pred[t+1] = np.tanh(
        #                    self.input_weight * np.matmul(self.W_in,
        #                    np.matmul(self.W_out,
        #                              np.hstack((self.r_pred[t], self.y_pred[t]))))
        #                + np.matmul(self.network, self.r_pred[t]))
        #                #update y:
        #                self.y_pred[t+1] = np.matmul(self.W_out,
        #                np.hstack((self.r_pred[t], self.y_pred[t])))
        #
        # print('extended_state = ', self.extended_states, ' worked')

        else:  # no extended states -> [r]
            self.r_pred[0] = self.activation_function(self.y_train[-1], self.r[-1])

            # transition from training to prediction
            self.y_pred[0] = np.matmul(self.W_out, self.r_pred[0])

            # prediction:
            for t in range(self.prediction_steps - 1):
                # update r:
                self.r_pred[t + 1] = self.activation_function(
                    self.y_pred[t] + self.noise[t],
                    self.r_pred[t])

                # update y:
                self.y_pred[t + 1] = np.matmul(self.W_out, self.r_pred[t + 1])

        # array (backtransform from sparse)
        self.network = self.network.toarray()

        t1 = time.time()
        if print_switch:
            print('predicting done in ', t1 - t0, 's')

    def calc_tt(self, flag='bool', split=0.1):
        """
        selects depending on if the abs(entry) of self.W_out is one of the
        largest, depending on split.
        If split is negative the abs(entry) smallest are selected depending
        on flag:
        - 'bool': self.W_out.shape with True/False 
        - 'bool_1d': is a projection to 1d
        - 'arg': returns args of the selection
        
        """
        absolute = int(self.N * split)

        n = self.ydim * self.N  # dof in W_out
        top_ten_bool = np.zeros(n, dtype=bool)  # False array
        arg = np.argsort(np.reshape(np.abs(self.W_out), -1))  # order of abs(W_out)
        if absolute > 0:
            top_ten_bool[arg[-absolute:]] = True  # set largest entries True
            top_ten_arg = np.argsort(np.max(np.abs(self.W_out), axis=0))[-absolute:]
        elif absolute < 0:
            top_ten_bool[arg[:-absolute]] = True  # set largest entries True
            top_ten_arg = np.argsort(np.max(np.abs(self.W_out), axis=0))[:-absolute]
        else:
            top_ten_arg = np.empty(0)

        top_ten_bool = np.reshape(top_ten_bool, self.W_out.shape)  # reshape to original shape
        top_ten_bool_1d = np.array(top_ten_bool.sum(axis=0), dtype=bool)  # project to 1d

        if flag == 'bool':
            return top_ten_bool
        elif flag == 'bool_1d':
            return top_ten_bool_1d
        elif flag == 'arg':
            return top_ten_arg

    def save_realization(self, filename='parameter/test_pickle_'):
        """
        Saves the network parameters (extracted with self.__dict__ to a file,
        for later reuse.
        
        pickle protocol version 2
        """
        self.timestamp = datetime.datetime.now()
        f = open(filename, 'wb')
        try:
            pickle.dump(self.__dict__, f, protocol=2)
        except:
            print('file could not be pickled: ', filename)
        f.close()

        return 'file saved'

    def load_realization(self, filename='parameter/test_pickle_', print_switch=False):
        """
        loads __dict__ from a pickled file and overwrites self.__dict__
        Original data lost!
        """
        g = open(filename, 'rb')
        try:
            dict_load = pickle.load(g)

            keys_init = self.__dict__.keys()
            keys_load = dict_load.keys()

            key_load_list = []
            key_init_list = []
            for key in keys_load:
                self.__dict__[key] = dict_load[key]  # this is where the dict is loaded
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

            return 'file loaded, __dict__ available in self.dict'
        except:
            print('file could not be unpickled: ', filename)
        g.close() 
    
    
        
#    def ft(self, traj):
#        ft_traj = np.zeros(shape=traj.shape, dtype=complex)
#        for dimension in np.arange(traj.shape[1]):
#            ft_traj[:,dimension] = np.fft.fft(traj[:,dimension])
#        
#        return ft_traj
#    
##    def max_lyapunov(self, tau_max=280, radius=0.3, plot=False):
##        #tau_max=280 for lorenz normal
##        #radius = 0.8
##
##        traj = self.y_pred
##        
##        t0 = time.time()
##        
##        if traj.shape[0] >= 50000:
##            print('traj large, creation of tree will take long')
##        #tau_max=400, radius=0.35 works        
##        """
##        Determines the largest Lyapunov exponent via Rosenstein-Kantz method.
##        See original papers of Rosenstein and Kantz.
##
##        parameters depend on self.dt and the system itself! 
##
##        traj: trajectory for which the maximal lyapunov exponent is determined
##        tau: timeshift after which one compares the bubble size
##        tau_max: maximal timeshift
##        s: sums (and averages) the all distances of former nearest neighbours
##            of a given point.
##        S: sums (and averages) the small s for all points as basis
##        """
##        #time steps for calculating the average distance:
##        taus = np.arange(100, tau_max, tau_max/13)
##        
##        #skip_points = 10
##        #number_basis_points = (traj.shape[0] - tau_max)/ skip_points
##        
##        #looking for the indices of next_neighbours for all points
##        tree = scipy.spatial.cKDTree(traj[:])
##        self.nn_index = tree.query_ball_point(traj, r=radius)
##
##        ###Drop points from close temporal environment        
##        #saving the number of nn for each point for histogram:
##        number_nn = []
##        points = 0
##        for point in np.arange(traj.shape[0]):
##            points += 1
##            number_nn.append(len(self.nn_index[point]))
##         
##        #drop all elements in nn_index lists where the neighbour is:
##        #1. less than 200 timesteps away
##        #2. where we cannot calculate the neighbours future in tau_max timesteps:
##        for point in np.arange(traj.shape[0]):
##            #print(point)            
##            self.nn_index[point] = [elem for elem in self.nn_index[point] if 
##                np.abs(elem - point) > 100 and elem + tau_max < traj.shape[0]]
##        
##        #same as above, after dropping the nn we are not interested in:
##        number_nn_ad = []
##        for point in np.arange(traj.shape[0]):
##            number_nn_ad.append(len(self.nn_index[point]))
##        #plot the number_nn distributions:
##        if plot:
##            plt.title('distributions of number of nn, check if there are enough nn after dropping')
##            plt.xlabel('# of nn')
##            plt.ylabel('# of points')
##            plt.hist(number_nn, label='nn_distr', bins=20, alpha=0.5)
##            plt.hist(number_nn_ad, label='nn_distr_ad', bins=20, alpha=0.5)
##            plt.legend()
##            plt.show()
##
##        #Calculate the largest Lyapunov exponent:
##        #for storing the results:
##        Sum = [] 
##        #loop over differnt tau, to get a functional dependence:
##        for tau in taus:
##            #print(tau)
##            
##            S = []#the summed values for all basis points 
##            
##            #loop over every point in the trajectory, where we can calclutate
##            #the future in tau_max timesteps:
##            for point in np.arange(traj.shape[0] - tau_max): #-1
##                #loop over nearest neighbours (if any) and average the distance of the
##                #evolved points after time tau        
##                if len(self.nn_index[point]) != 0:      
##                    s = [] #the running sum over distances for one point
##                
##                    for index in self.nn_index[point]:               
##                        s.append(np.linalg.norm(traj[point+tau] - traj[index + tau]))
##                    s = np.array(s).mean()
##                    S.append(np.log(s)) #add one points average s to S
##                    
##                    #since there is no else, we only avg over the points that have
##                    #points in their epsilon environment
##            Sum.append((tau*self.dt, np.array(S).mean()))
##        self.Sum = np.array(Sum)
##        
##        #validation plot:
##        slope, intercept = np.polyfit(self.Sum[:,0], self.Sum[:,1], deg=1)
##        
##        t1= time.time()
##        #print(t1-t0, 'time Lyapunov exp')
##        if plot:
##            plt.title('slope: ' + str(slope) +', intercept: ' + str(intercept))
##            plt.xlabel('tau')
##            plt.ylabel('< log(dist. former nn) >')
##            plt.plot(self.Sum[:,0], self.Sum[:,1])
##            plt.plot(self.Sum[:,0], self.Sum[:,0]*slope + intercept)
##            plt.show()
##        
##        return slope
##    
##    def max_lyapunov_fast(self, traj, tau_min=120,tau_max=260, radius=0.8,
##                          plot=False):
##        t0 = time.time()
##        taus = np.array([tau_min, tau_max])
##        
##        tree = scipy.spatial.cKDTree(traj[:])
##        self.nn_index = tree.query_ball_point(traj, r=radius)
##        ###Drop points from close temporal environment
##        #saving the number of nn for each point for histogram:
##        number_nn = []
##        points = 0
##        for point in np.arange(traj.shape[0]):
##            points += 1
##            number_nn.append(len(self.nn_index[point]))
##         
##        #drop all elements in nn_index lists where the neighbour is:
##        #1. less than 200 timesteps away
##        #2. where we cannot calculate the neighbours future in tau_max timesteps:
##        for point in np.arange(traj.shape[0]):
##            #print(point)            
##            self.nn_index[point] = [elem for elem in self.nn_index[point] if 
##                np.abs(elem - point) > 100 and elem + tau_max < traj.shape[0]]
##        
##        #same as above, after dropping the nn we are not interested in:
##        number_nn_ad = []
##        for point in np.arange(traj.shape[0]):
##            number_nn_ad.append(len(self.nn_index[point]))
##            
##        #plot the number_nn distributions:
##        if plot:
##            plt.title('distributions of number of nn, check if there are enough nn after dropping')
##            plt.xlabel('# of nn')
##            plt.ylabel('# of points')
##            plt.hist(number_nn, label='nn_distr', bins=20, alpha=0.5)
##            plt.hist(number_nn_ad, label='nn_distr_ad', bins=20, alpha=0.5)
##            plt.legend()
##            plt.show()
##
##        #Calculate the largest Lyapunov exponent:
##        #for storing the results:
##        Sum = []
##        #loop over differnt tau, to get a functional dependence:
##        for tau in taus:
##            #print(tau)
##            
##            S = []#the summed values for all basis points 
##            
##            #loop over every point in the trajectory, where we can calclutate
##            #the future in tau_max timesteps:
##            for point in np.arange(traj.shape[0] - tau_max): #-1
##                #loop over nearest neighbours (if any) and average the distance of the
##                #evolved points after time tau        
##                if len(self.nn_index[point]) != 0:      
##                    s = [] #the running sum over distances for one point
##                
##                    for index in self.nn_index[point]:               
##                        s.append(np.linalg.norm(traj[point+tau] - traj[index + tau]))
##                    s = np.array(s).mean()
##                    S.append(np.log(s)) #add one points average s to S
##                    
##                    #since there is no else, we only avg over the points that have
##                    #points in their epsilon environment
##            Sum.append((tau*self.dt, np.array(S).mean()))
##        self.Sum = np.array(Sum)
##        
##        
##        
##        slope = (self.Sum[-1,-1] - self.Sum[0,-1]) / (self.Sum[-1,0] - self.Sum[0,0])
##        
##        t1 = time.time()
##        #print('time: ', t1-t0)        
##        
##        self.lyapunov_exponent = slope
##        
##        #return slope
#    

#        
#    def calc_chi(self):
#        traj = self.y_pred
#        """
#        max_lya: maximal Lyapunov exponent
#        C_dim: correlation dimension
#        <-->, std(--): expectation value, standard deviation for 1000 original
#            trajectories
#        chi = max_lya - <max_lya> / std(max_lya) + C_dim - <C_dim> / std(C_dim)
#        """
#        #exp and std for 100 trajectories a 25k points:
#        exp_max_lya = 0.82231800951535061
#        std_max_lya = 0.019966305991885182
#        exp_C_dim = 2.0307003809606998
#        std_C_dim = 0.016888285378633988
#        
#        self.chi = (self.max_lyapunov_nn(traj) - exp_max_lya )/std_max_lya + \
#            (self.correlation_dimension(traj) - exp_C_dim )/std_C_dim
#            
#        #return self.chi
#        
#    def autocorrelation_zero_crossing(self, signal):
#        """
#        returns the timestep at which the signal (1d. array) has the first zero
#        crossing.
#        used for embedding correctly
#        """
#        #signal = self.y_train        
#        interval = signal.shape[0]
#        ac = np.zeros(interval)
#        
#        for tau in np.arange(interval):
#            
#            
#            for t in np.arange(interval):
#                ac[tau] += signal[t] \
#                    * signal[(t+tau) % interval]
#            if np.sign(ac[0]*ac[tau]) == -1.:
#                return tau
#                break
#    

#    def do_complete_network(self):
#        """
#        If extended states (not only the reservoir states are used for fitting
#        W_out, but also the input x) are used, the adjacency matrix is
#        completed to depict this, by appending a unit matrix on the diagonal
#        """
#        if self.extended_states:
#            #larger adjacency matrix is initialized:
#            self.complete_network = np.zeros((self.N + self.xdim,
#                                          self.N + self.xdim))
#            #and filled with the old adj.mat and the unit matrix:
#            self.complete_network[:self.N, :self.N] = self.network
#            #maybe zero instead of eye is suitable: 
#            self.complete_network[self.N: ,self.N:] = np.eye(self.xdim)
#        
#        else:
#            self.complete_network = self.network
#                
#    def draw_network(self):
#        """"
#        Shows a plot of the network plus the direct input
#        
#        want to colorcode the nodes that are most connected to the output
#        (through all three dimensions)
#        hopefully, one can see some connection to structure in the network
#        """        
#        self.do_complete_network()
#        graph = nx.from_numpy_matrix(self.complete_network)
#        
#        labels = dict([ (i,i) for i in range(self.complete_network.shape[0])])
#        plt.figure(figsize = (12,12))
#        nx.draw(graph, nx.circular_layout(graph), node_color = 'r',
#                node_size = 30, alpha = 0.5, labels=labels)
#        nx.draw_networkx_nodes(graph, nx.circular_layout(graph),
#                               node_color='darkgreen', nodelist=self.top_ten,
#                               node_size=200, alpha=0.8)
#        plt.show()
#        """
#        since we added eye to the complete_network the in/out_degree counts
#        differ from the drawn graph, bc. the connection to a node itelf is not
#        displayed but counted!
#        """
#        
#    def network_measures(self):
#        """
#        Uses the network to calculate the degree distribution in two ways
#        """
#        '''
#        => use in/out degree from adj. matrix instead of nx.degree
#        requires W_out from train()
#        '''        
#        ### in/out_degree
#        self.calc_binary_network()
#        #self.do_complete_network()
#        degree = nx.degree_histogram(nx.from_numpy_matrix(self.network))
#        #print(degree)       
#        plt.plot(np.arange(len(degree)), degree, 'x')
#        plt.title('nx.degree distribution plot of the nextwork')        
#        plt.loglog()
#        plt.show()
#        top_ten = list(np.argsort(-np.sum(self.W_out, axis = 0))[:10])
#        print(nx.degree(nx.from_numpy_matrix(self.network), top_ten))
#        #nx.to_numpy_matrix: edges from i to j, become entries at A[i,j]
#        #-> row_sums (axis=1) are out_degree
#        in_degree = self.binary_network.sum(axis=0)
#        out_degree = self.binary_network.sum(axis=1)
#        plt.hist(in_degree, bins=15, label='in_degree', alpha=0.5, log=True, histtype='stepfilled')
#        plt.hist(out_degree, bins=15, label='out_degree', alpha=0.5, log=True, histtype='stepfilled')    
#        plt.legend()
#        plt.title('in/out_degree distribution plot of the network')
#        plt.show()
#        #print(in_degree[top_ten], out_degree[top_ten])
#        

#            
#    def exp_dev(self):
#        """
#        calculate the expectation value and std deviation for i trajectories
#        with length 25000 of lyapunov exp and correlation dim for y_train
#        sum(axis=0).mean()
#        """
#        i = 100
#        self.lya = np.zeros(i)
#        self.C_dim = np.zeros(i)
#        for j in np.arange(0, 25000*i, 25000):
#            print(j/25000, 'of 100')
#            traj = self.y_train[j:j+25000]
#            self.lya[j/25000] = self.max_lyapunov_nn(traj)
#            self.C_dim[j/25000] = self.correlation_dimension(traj)
#            print(traj[0,0])
#            #self.load_data()
#            

#            
#          
##if __main__:
#
#### INLINE COMMANDS
##==============================================================================
##A = reservoir(number_of_nodes=500, input_dimension=3,
##                 output_dimension=3, type_of_network='random',
##                 dt = 2e-2, training_steps=5000,
##                 prediction_steps=5000, discard_steps=5000,
##                 regularization_parameter=0.0001, spectral_radius=1.5,
##                 input_weight=.7, avg_degree=6., epsilon=np.array([5,10,5]),
##                 extended_states=False)
##A.vary_network()
##A.load_data()
##A.train()
##A.predict()
##A.RMSE('pred')
##A.plt('t')
#
