"""
:author: aumeier
:date: 2019/04/26
:license: ???
"""

import scipy.signal
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import time
import datetime
import copy

import lorenz

class reservoir(object):
    def __init__(self, system=lorenz.mod_lorenz, number_of_nodes=100, input_dimension=3,
                 output_dimension=3, type_of_network='small_world',
                 dt = 2e-2, training_steps=10000,
                 prediction_steps=10000, discard_steps=10000,
                 regularization_parameter=0.0001, spectral_radius=0.5,
                 input_weight=1., avg_degree=6., epsilon=np.array([5,10,5]),
                 extended_states=False, W_in_sparse=True, W_in_scale=1.):
        """
        reservoir is a class for reservoir computing, using different network
        structures to predict chaotic time series
          
        :parameters:
        :system: choose from lorenz.lorenz, lorenz.mod_lorenz and 
            lorenz.roessler
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
            in the cost function
        :spectral_radius: spectral radius of the actual reservoir network
        :input_weight: weight of the input state over the reservoir state in
            the neurons
        :edge_prob: the probability with which an edge is created in a random
            network
        :epsilon: threshold distance between y_test and y_pred for the
            prediction to be consided close
        :extended_state: bool: if True: an extended state of the form [b,x,r]
            is used for updating the reservoir state, the prediction and
            fitting W_out.
            if False: only [r] is used 
        """
        self.system = system
        self.N = number_of_nodes
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
        self.avg_degree = float(avg_degree)# = self.network.sum()/self.N
        self.edge_prob = self.avg_degree/ (self.N-1)
        self.b_out = np.ones((self.training_steps,1)) #bias in fitting W_out
        self.extended_states = extended_states
        self.epsilon = epsilon
        self.W_in_sparse = W_in_sparse
        self.W_in_scale = W_in_scale
        
        #topology of the network, adjacency matrix with entries 0. or 1.:
        self.binary_network = None # calc_binary_network() assigns values
        
        #load_data() assigns values to:        
        self.x_train = None
        self.x_discard = None
        self.y_train = None
        self.y_test = None 
        self.W_in = None

        #train() assigns values to:
        self.W_out = None        
        self.r = None

        #predict() assigns values to:
        self.r_pred = None
        self.y_pred = None
        self.noise = None
        
        #methods for analyizing assing values to:
        self.timestamp = None #timestamp; value assigned in save_realization()
        self.complete_network = None #complete_network() assigns value
        self.demerge_time = None #demerge_time() assigns value
        self.lyapunov_exponent = None #max_lyapunov_nn() assigns value  
        self.lyapunov_exponent_nn = None        
        self.lyapunov_exponent_nn_test = None 
        self.dimension = None #correlation_dimension(traj) assigns value
        self.dimension_test = None
        self.chi = None #calc_chi
        self.in_strength = None
        self.avg_in_strength = None
        self.out_strength = None
        self.avg_out_strength = None
        self.avg_weighted_cc = None
        self.weighted_cc = None
        self.clustering_coeff = None
        self.avg_clustering_coeff = None
        self.top_ten_arg_1d = None
        self.top_ten_bool = None
        self.top_ten_bool_1d = None
        
        
        #network type flags are handled:
        if type_of_network == 'random':
            network = nx.fast_gnp_random_graph(self.N, self.edge_prob)
            
        elif type_of_network == 'scale_free':
            network = nx.barabasi_albert_graph(self.N, int(self.avg_degree/2))
            #radius, dimension have to be specified
        elif type_of_network == 'small_world':
            network = nx.watts_strogatz_graph(self.N, k=int(self.avg_degree), p=0.1)
        """
        elif type_of_network == 'random_geometic':
            network = nx.random_geometric_graph(self.N)       
        """
        #make a numpy array out of the network's adjacency matrix:
        
        self.network = np.asarray(nx.to_numpy_matrix(network))
        
        
        
        """
        PROBLEM: when initialized this is not executed properly!
        """
        #contains scaling with spectral_radius:
        #self.vary_network()
        self.calc_binary_network()
        
    def calc_binary_network(self):
        """
        copy.copy resolves the problem, I don't get why it is necessary
        -> ask Daniel
        #self.binary_network = copy.copy(self.network)        
        """
        self.binary_network = np.zeros((self.network.shape))
        self.binary_network[np.nonzero(self.network)] = 1.

    def vary_network(self, print_switch=False):
        """
        Varies the weights, while conserving the topology.
        The non-zero elements of the adjacency matrix are uniformly randomized,
        and the matrix is normalized.
        
        When doing paramter scans, we want to vary only the network weights,
        but not the overall shape, nor the W_in matrix.
        W_out has to be fitted for every realization.
        """
        """
        set all measures and derivatives to zero or None
        what about binary_network?
        keep W_in
        """
        #print('test_1: ', np.unique(self.network).shape)
        t0 = time.time()
        
        #contains tuples of non-zero elements:
        arg_binary_network = np.argwhere(self.network)
        
        #uniform entries from [-0.5, 0.5) at the former non-zero locations:
        self.network[arg_binary_network[:,0],
                     arg_binary_network[:,1]] = np.random.random(
                     size=self.network[self.network != 0.].shape)-0.5

        self.scale_network()
        
        t1 = time.time()
        #print('test_2: ', np.unique(self.network).shape)
        if print_switch:
            print('varied non-zero entries in network in ', t1-t0, 's')
            
    def scale_network(self):
        #scale adjacency matrix, according to desired spectral radius:
        self.network = self.spectral_radius*(self.network/np.absolute(
            np.linalg.eigvals(self.network)).max())
            
    def load_data(self, add_noise=False, std_noise=0.1, print_switch=False,
                  starting_point=None, fix_start=False, start_from_attractor=False):
        """
        Method to load data from a file for training and testing the network.
        If the file does not exist, it is created according to the parameters.
        W_in is filled with one non-zero element per column at random.
        
        :parameters:
        :add_noise: boolean: if normal distributed noise should be added to the
            imported timeseries
        :std_noise: standard deviation of the applied noise
        """   
        t0 = time.time()
        
        #minimum size for vals, lorenz.save_trajectory has to evaluate:
        timesteps = 1 + self.discard_steps + self.training_steps \
                    + self.prediction_steps
        
        ### Saves complete (discard, train, pred) trajectory to txt file:        
        #==============================================================================
        #         filename = 'lorenz_attractor_'+str(timesteps)+'_a_'+str(self.dt)+'.dat'        
        #         """        
        #         try:
        #             vals = np.loadtxt(filename)
        #             print('data loaded from file <' + filename + '>')    
        #         except:
        #         """    
        #         ###           
        #         lorenz.save_trajectory(timesteps=timesteps, dt=self.dt)    
        #         #print('no file found')            
        #         vals = np.loadtxt(filename)
        #         #print('file <' + filename + '> created, data loaded from file')
        #         ###        
        #==============================================================================
        ### Only passes the vals, saving via save_realization:
             
        if start_from_attractor:
            length = 100000
            original_start = np.array([-2.00384153, -5.34877257, -1.20401106])
            random_index = np.random.choice(np.arange(2000, 10000, 1))
            if print_switch:
                print('random index for starting_point: ', random_index)
            starting_point = lorenz.simulate_trajectory(f=self.system, dt=self.dt, timesteps=length + 1000,
                                                        starting_point=original_start, fix_start=True)[random_index]
            vals = lorenz.simulate_trajectory(f=self.system, dt=self.dt, timesteps=timesteps,
                                              starting_point=starting_point, fix_start=True)
        else:   
        
            vals = lorenz.simulate_trajectory(f=self.system, dt=self.dt, timesteps=timesteps,
                                              starting_point=starting_point, fix_start=fix_start)
                
        #print('data loading successfull')
        if add_noise:
            vals += np.random.normal(scale=std_noise, size=vals.shape)
            print('added noise with std_dev: ' + str(std_noise))
        #define local variables for test/train split:
        n_test = self.prediction_steps
        n_train = self.training_steps + self.discard_steps
        n_discard = self.discard_steps
            
        #sketch of test/train split:
        #[--n_discard --|--n_train--|--n_test--]
        #and y_train is shifted by 1
        self.x_train = vals[n_discard:n_train,:] #input values driving reservoir
        self.x_discard = vals[:n_discard,:]
        self.y_train = vals[n_discard+1:n_train+1,:] #+1
        self.y_test = vals[n_train+1:n_train+n_test+1,:] #+1
        
        if self.W_in_sparse:
            #W_in such that one element in each row is non-zero (Lu,Hunt, Ott 2018):
            #self.W_in = np.random.uniform(low=-1.,high=1.,(self.N, self.xdim))
            self.W_in = np.zeros((self.N,self.xdim))            
            for i in range(self.N):
                random_x_coord = np.random.choice(np.arange(self.xdim))
                self.W_in[i, random_x_coord] = np.random.uniform(
                    low=-self.W_in_scale, high=self.W_in_scale) #maps input values to reservoir
        else:
            self.W_in = np.random.uniform(low=-self.W_in_scale,
                                          high=self.W_in_scale,
                                          size=(self.N,self.xdim))
        
        t1 = time.time()
        if print_switch:
            print('input (x) and target (y) loaded in ', t1-t0, 's')            
    def normalize_data(self):
        """
        normalizes the input of all dimensions respectivley, by using 
        (min, max) of x_discard
        neg values to -1
        pos values to 1
        -> asymmetric scaling -> test
        """
        if self.system == lorenz.roessler:
            self.minima = self.x_discard.min(axis=0)
            self.maxima = self.x_discard.min(axis=0)
            for x in [self.x_discard, self.x_train, self.y_test, self.y_train]:
                for dim in np.arange(self.xdim):
                    """x[:,dim] = here comes the scaling!"""
            
        else:
            print('normalization only available for roessler system')
        
        
    def train(self, print_switch=False):
        '''
        Fits W_out, which connects the reservoir states and the input to the
        desired output, using linear regression and Tikhonov regularization.
        Discards discard_steps steps befor recording r, to synchronize the
        network with the input.
        Requires load_data() first, to pass values to x_train, y_train, y_test
        '''
        t0 = time.time()
        
        #states of the reservoir:
        self.r = np.zeros((self.training_steps, self.N))
                    
        #reservoir is synchronized with trajectory during discard_steps:            
        for t in range(self.discard_steps):

            self.r[0] = np.tanh(
                self.input_weight *
                np.matmul(self.W_in, self.x_discard[t]) + \
                np.matmul(self.network, self.r[0]) )
        
        #print('discarding done')
        
        #states are then used to fit the target y_train:
        
        for t in range(self.training_steps-1):
            self.r[t+1] = np.tanh(
               self.input_weight *np.matmul(self.W_in, self.x_train[t+1]) + \
               np.matmul(self.network, self.r[t]) )#vector equation with
               # self.N entries
        #print('iterating training steps done')
        #calculating P from r and x, watch out:
        #in my convention the first index is for time, in the following
        #we use transposed notation to match the convention in
        #zimmermann, parlitz
        '''
        zimmermann, parlitz paper uses not only the reservoir states but also
        the current input for predicting an outcome
        the x(input) values are appended to the r(reservoir state) values
        transpose to match the formula from the paper
        W_out will be used as follows: y[t+1] = W_out*r[t]
        '''
#==============================================================================
#             print('b_out_dim: ', self.b_out.shape)
#             print('r_train_dim: ', self.r.shape)
#             print('x_train_dim: ', self.x_train.shape)
#==============================================================================
        """
        timing for extended_states not adapted to new_timing yet        
        """
        """        
        if self.extended_states:
            #create the extended state:[b_out, r, x] and time as 2nd dimension:
            #X = np.concatenate((self.b_out, self.r, self.x_train), axis=1).T
            #create the extended state:[r, x] and time as 2nd dimension:
            X = np.concatenate((self.r, self.x_train), axis=1).T
        
        else:
        """
        X = self.r.T

            

        Y = self.y_train.T
        #print('begin to calc W_out')
        self.W_out = np.matmul(
            np.matmul(Y,X.T), np.linalg.inv(np.matmul(X,X.T)
            + self.reg_param*np.eye(X.shape[0])))
        
        #calc top ten removed, since it disturbs loops in remove_nodes_statistical.py
        
        t1 = time.time()
        if print_switch:
            print('training done in ', t1-t0, 's')
        
    def predict(self, print_switch=False, prediction_noise=False, noise_scale=0.1):
        """
        Uses the trained network to predict output, using the network as
        recurrent network, feeding back in the (noisy) output.
        """
        t0 = time.time()
    
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
            """
            should I implement b as well?
            """
            #create the extended state:[b_out, r, x] and time as 2nd dimension:
            #X = np.concatenate((self.b_out, self.r, self.x_train), axis=1).T
            
            #create the extended state:[r, x] and time as 2nd dimension:
            
            self.y_pred[0] = np.matmul(self.W_out,
                                np.hstack((self.r[-1], self.x_train[-1])))
            
            for t in range(self.prediction_steps - 1):
                """
                Add noise
                """
                #update r:
                self.r_pred[t+1] = np.tanh(
                    self.input_weight * np.matmul(self.W_in,
                    np.matmul(self.W_out,
                              np.hstack((self.r_pred[t], self.y_pred[t]))))
                + np.matmul(self.network, self.r_pred[t]))
                #update y:
                self.y_pred[t+1] = np.matmul(self.W_out,
                np.hstack((self.r_pred[t], self.y_pred[t])))
        
            #print('extended_state = ', self.extended_states, ' worked')
        
        else: #no extended states -> [r]
            self.r_pred[0] = np.tanh(self.input_weight *
                    np.matmul(self.W_in, self.y_train[-1])
                    + np.matmul(self.network, self.r[-1]))
            #transition from training to prediction
                    
            self.y_pred[0] = np.matmul(self.W_out, self.r_pred[0])
            
            #prediction:
            for t in range(self.prediction_steps - 1):
                #update r:
                self.r_pred[t+1] = np.tanh(self.input_weight *
                    np.matmul(self.W_in, self.y_pred[t] + self.noise[t])
                    + np.matmul(self.network, self.r_pred[t]))
                #update y:
                self.y_pred[t+1] = np.matmul(self.W_out, self.r_pred[t+1])

        t1 = time.time()
        if print_switch:
            print('predicting done in ', t1-t0, 's')
            
    def calc_top_ten(self, split=0.1):
        """
        top_ten_bool: W_out with True/False entries depending on if the abs(entry)
        is one of the largest, depending on split.
        if split is negative the abs(entry) smallest are selected.
        top_ten_arg: split of the max_over_dims(abs(W_out))
        """
        absolute = int(self.N * split)
        
        n = self.W_out.shape[0]*self.W_out.shape[1] #dof in W_out
        self.top_ten_bool = np.zeros(n, dtype=bool) #False array
        arg = np.argsort(np.reshape(np.abs(self.W_out), -1)) #order of abs(W_out)
        if absolute > 0:        
            self.top_ten_bool[arg[-absolute:]] = True #set largest entries True
            self.top_ten_arg = np.argsort(np.max(np.abs(self.W_out), axis=0))[-absolute:]
        if absolute < 0:
            self.top_ten_bool[arg[:-absolute]] = True #set largest entries True
            self.top_ten_arg = np.argsort(np.max(np.abs(self.W_out), axis=0))[:-absolute]
        self.top_ten_bool = np.reshape(self.top_ten_bool, self.W_out.shape) #reshape to original shape
        self.top_ten_bool_1d = np.array(self.top_ten_bool.sum(axis=0), dtype=bool) #project to 1d
        #self.top_ten_arg_1d = np.arange(self.N)[self.top_ten_bool_1d] #node number (1d)
        
             
        
    def RMSE(self, flag):
        """
        Measures an average over the spatial distance of predicted and actual
        trajectory
        """
        if flag == 'train':
            error = np.sqrt(((np.matmul(self.W_out, self.r.T) - \
                                        self.y_train.T)**2).sum() \
                    / (self.y_train**2).sum())
            self.RMSE_train = error
        elif flag == 'pred':
            error = np.sqrt(((np.matmul(self.W_out, self.r_pred.T) - \
                                        self.y_test.T)**2).sum() \
                    / (self.y_test**2).sum())
            self.RMSE_pred = error
        else:
            print('use "train" or "pred" as flag')
        
    def calc_demerge_time(self):
        """
        Measure for the quality of the predicted trajectory
        Number of timesteps it takes for the prediction to loose track of the real
        trajectory.
        Returns the number of steps for which y_test and y_pred are separated less
        than epsilon in each dimension.
        """
        delta = np.abs(self.y_test - self.y_pred)
        self.demerge_time = np.argmax(delta > self.epsilon, axis=0).min()
        """
        true_false_array = np.array([np.abs(self.y_test - self.y_pred) < self.epsilon])[0]
       
        #print(self.true_false_array)
        for T in np.arange(true_false_array.shape[0]):
            for i in np.arange(true_false_array.shape[1]):

                if true_false_array[T,i] == False:
                    self.demerge_time = T
                    return T
                    break
            else:
                continue
            break
        """
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
        loads __dict__ from a file and overwrites self.__dict__
        Original data lost!
        """

        g = open(filename, 'rb')
        try:
            dict_load= pickle.load(g)
            
            keys_init = self.__dict__.keys()
            keys_load = dict_load.keys()
            
            key_load_list = []
            key_init_list = []
            for key in keys_load:
                self.__dict__[key] = dict_load[key] #this is where the dict is loaded
                print(str(key)+' was loaded to __dict__')                
                if not key in keys_init:
                    key_load_list.append(key)
                    #print(key)                    
                    
            for key in keys_init:
                if not key in keys_load:
                    key_init_list.append(key)
                    #print(key)                    
            for key_init in key_init_list:
                print('the following attributes and methonds are missing in the' +
                    ' loaded reservoir: ' + str(key_init) + str(keys_init[key_init])[:10])
            for key_load in key_load_list:
                print('the following attributes and methonds are missing in the' +
                    ' initialized reservoir: ' + str(key_load))
            
        except:
            print('file could not be unpickled: ', filename)
        g.close()

        return 'file loaded, __dict__ available in self.dict'

    def return_map(self, axis=2):
        """
        Shows the recurrence plot of the maxima of a given axis
        """
        max_pred = self.y_pred[scipy.signal.argrelextrema(self.y_pred[:,2],
            np.greater, order = 5),axis]
        max_test = self.y_test[scipy.signal.argrelextrema(self.y_test[:,2],
            np.greater, order=5),axis]
        plt.plot(max_pred[0,:-1:2], max_pred[0,1::2],
                 '.', color='red', alpha=0.5, label='predicted y')
        plt.plot(max_test[0,:-1:2], max_test[0,1::2],
                 '.', color='green', alpha=0.5, label='test y')
        plt.legend(loc=2, fontsize=10)
        plt.show()

    def correlation_dimension(self, r_min=0.5, r_max=5., r_steps=0.15,
                              plot=False, test_measure=False):
        """
        traj: trajectory of an attractor, whos correlation dimension is returned        
        First we calculate a sum over all points within a given radius, then
        average over all basis points and vary the radius
        (grassberger, procaccia).
        
        parameters depend on self.dt and the system itself!
        
        N_r: list of tuples: (radius, average number of neighbours within all
            balls)
        self.dimension: slope of the log.log plot assumes:
            N_r(radius) ~ radius**dimension
        """
        
        if test_measure:
            traj = self.y_train # for measure assessing
        else:
            traj = self.y_pred # for evaluating prediction
        
        #save time by using only half the points:
        #traj = traj[::2]
    
        t0 = time.time()
        
        nr_points = float(traj.shape[0])
        radii = np.arange(r_min, r_max, r_steps)
        
        tree = scipy.spatial.cKDTree(traj)
        N_r = np.array(tree.count_neighbors(tree, radii), dtype=float)/ nr_points
        N_r = np.vstack((radii, N_r))
        
        #linear fit based on loglog scale, to get slope/dimension:
        slope, intercept = np.polyfit(np.log(N_r[0]), np.log(N_r[1]), deg=1)
        if test_measure:
            self.dimension_test = slope
        else:
            self.dimension = slope

        t1 = time.time()
        
        ###plotting
        if plot:        
            plt.loglog(N_r[0], N_r[1], 'x', basex=10., basey=10.)
            plt.title('loglog plot of the N_r(radius), slope/dim = '+ str(slope))
            plt.show()        
                

        #print(t1-t0, ': time for dimension w/ plotting')
        
        #return self.dimension
        
    def ft(self, traj):
        ft_traj = np.zeros(shape=traj.shape, dtype=complex)
        for dimension in np.arange(traj.shape[1]):
            ft_traj[:,dimension] = np.fft.fft(traj[:,dimension])
        
        return ft_traj
    
    def max_lyapunov(self, tau_max=280, radius=0.3, plot=False):
        #tau_max=280 for lorenz normal
        #radius = 0.8

        traj = self.y_pred
        
        t0 = time.time()
        
        if traj.shape[0] >= 50000:
            print('traj large, creation of tree will take long')
        #tau_max=400, radius=0.35 works        
        """
        Determines the largest Lyapunov exponent via Rosenstein-Kantz method.
        See original papers of Rosenstein and Kantz.

        parameters depend on self.dt and the system itself! 

        traj: trajectory for which the maximal lyapunov exponent is determined
        tau: timeshift after which one compares the bubble size
        tau_max: maximal timeshift
        s: sums (and averages) the all distances of former nearest neighbours
            of a given point.
        S: sums (and averages) the small s for all points as basis
        """
        #time steps for calculating the average distance:
        taus = np.arange(100, tau_max, tau_max/13)
        
        #skip_points = 10
        #number_basis_points = (traj.shape[0] - tau_max)/ skip_points
        
        #looking for the indices of next_neighbours for all points
        tree = scipy.spatial.cKDTree(traj[:])
        self.nn_index = tree.query_ball_point(traj, r=radius)

        ###Drop points from close temporal environment        
        #saving the number of nn for each point for histogram:
        number_nn = []
        points = 0
        for point in np.arange(traj.shape[0]):
            points += 1
            number_nn.append(len(self.nn_index[point]))
         
        #drop all elements in nn_index lists where the neighbour is:
        #1. less than 200 timesteps away
        #2. where we cannot calculate the neighbours future in tau_max timesteps:
        for point in np.arange(traj.shape[0]):
            #print(point)            
            self.nn_index[point] = [elem for elem in self.nn_index[point] if 
                np.abs(elem - point) > 100 and elem + tau_max < traj.shape[0]]
        
        #same as above, after dropping the nn we are not interested in:
        number_nn_ad = []
        for point in np.arange(traj.shape[0]):
            number_nn_ad.append(len(self.nn_index[point]))
        #plot the number_nn distributions:
        if plot:
            plt.title('distributions of number of nn, check if there are enough nn after dropping')
            plt.xlabel('# of nn')
            plt.ylabel('# of points')
            plt.hist(number_nn, label='nn_distr', bins=20, alpha=0.5)
            plt.hist(number_nn_ad, label='nn_distr_ad', bins=20, alpha=0.5)
            plt.legend()
            plt.show()

        #Calculate the largest Lyapunov exponent:
        #for storing the results:
        Sum = [] 
        #loop over differnt tau, to get a functional dependence:
        for tau in taus:
            #print(tau)
            
            S = []#the summed values for all basis points 
            
            #loop over every point in the trajectory, where we can calclutate
            #the future in tau_max timesteps:
            for point in np.arange(traj.shape[0] - tau_max): #-1
                #loop over nearest neighbours (if any) and average the distance of the
                #evolved points after time tau        
                if len(self.nn_index[point]) != 0:      
                    s = [] #the running sum over distances for one point
                
                    for index in self.nn_index[point]:               
                        s.append(np.linalg.norm(traj[point+tau] - traj[index + tau]))
                    s = np.array(s).mean()
                    S.append(np.log(s)) #add one points average s to S
                    
                    #since there is no else, we only avg over the points that have
                    #points in their epsilon environment
            Sum.append((tau*self.dt, np.array(S).mean()))
        self.Sum = np.array(Sum)
        
        #validation plot:
        slope, intercept = np.polyfit(self.Sum[:,0], self.Sum[:,1], deg=1)
        
        t1= time.time()
        #print(t1-t0, 'time Lyapunov exp')
        if plot:
            plt.title('slope: ' + str(slope) +', intercept: ' + str(intercept))
            plt.xlabel('tau')
            plt.ylabel('< log(dist. former nn) >')
            plt.plot(self.Sum[:,0], self.Sum[:,1])
            plt.plot(self.Sum[:,0], self.Sum[:,0]*slope + intercept)
            plt.show()
        
        return slope
    
    def max_lyapunov_fast(self, traj, tau_min=120,tau_max=260, radius=0.8,
                          plot=False):
        t0 = time.time()
        taus = np.array([tau_min, tau_max])
        
        tree = scipy.spatial.cKDTree(traj[:])
        self.nn_index = tree.query_ball_point(traj, r=radius)
        ###Drop points from close temporal environment
        #saving the number of nn for each point for histogram:
        number_nn = []
        points = 0
        for point in np.arange(traj.shape[0]):
            points += 1
            number_nn.append(len(self.nn_index[point]))
         
        #drop all elements in nn_index lists where the neighbour is:
        #1. less than 200 timesteps away
        #2. where we cannot calculate the neighbours future in tau_max timesteps:
        for point in np.arange(traj.shape[0]):
            #print(point)            
            self.nn_index[point] = [elem for elem in self.nn_index[point] if 
                np.abs(elem - point) > 100 and elem + tau_max < traj.shape[0]]
        
        #same as above, after dropping the nn we are not interested in:
        number_nn_ad = []
        for point in np.arange(traj.shape[0]):
            number_nn_ad.append(len(self.nn_index[point]))
            
        #plot the number_nn distributions:
        if plot:
            plt.title('distributions of number of nn, check if there are enough nn after dropping')
            plt.xlabel('# of nn')
            plt.ylabel('# of points')
            plt.hist(number_nn, label='nn_distr', bins=20, alpha=0.5)
            plt.hist(number_nn_ad, label='nn_distr_ad', bins=20, alpha=0.5)
            plt.legend()
            plt.show()

        #Calculate the largest Lyapunov exponent:
        #for storing the results:
        Sum = []
        #loop over differnt tau, to get a functional dependence:
        for tau in taus:
            #print(tau)
            
            S = []#the summed values for all basis points 
            
            #loop over every point in the trajectory, where we can calclutate
            #the future in tau_max timesteps:
            for point in np.arange(traj.shape[0] - tau_max): #-1
                #loop over nearest neighbours (if any) and average the distance of the
                #evolved points after time tau        
                if len(self.nn_index[point]) != 0:      
                    s = [] #the running sum over distances for one point
                
                    for index in self.nn_index[point]:               
                        s.append(np.linalg.norm(traj[point+tau] - traj[index + tau]))
                    s = np.array(s).mean()
                    S.append(np.log(s)) #add one points average s to S
                    
                    #since there is no else, we only avg over the points that have
                    #points in their epsilon environment
            Sum.append((tau*self.dt, np.array(S).mean()))
        self.Sum = np.array(Sum)
        
        
        
        slope = (self.Sum[-1,-1] - self.Sum[0,-1]) / (self.Sum[-1,0] - self.Sum[0,0])
        
        t1 = time.time()
        #print('time: ', t1-t0)        
        
        self.lyapunov_exponent = slope
        
        #return slope
    
    def max_lyapunov_nn(self, threshold=int(10),
                        plot=False, print_switch=False, test_measure=False):
        """
        Calculates the maximal Lyapunov Exponent from traj, by estimating the
        time derivative of the mean logarithmic distances of former next
        neighbours. Only values for tau_min/max are used for calculating the
        slope!
        
        Since the attractor has a size of roughly 20 [log(20) ~ 3.] this
        distance reaches a maximum after a certain time, approximately
        after 4. time units [time_units = dt*steps]
        Therefore the default values are choosen to be dt dependent as in 
        ###Definition of taus:
           
        tau_min/max are given in units of steps
        plot to check for correct average
        """
        """
        REMINDER:
        remove the loop over taus, since the slope is calculated with single
        values only
        """
        t0 = time.time()
        
        ###Definition of taus:        
        tau_min=int(0.5/self.dt)        
        tau_max=int(3.8/self.dt)
        taus = np.arange(tau_min, tau_max, 10) #taus = np.array([tau_min, tau_max])
        
        if test_measure:
            traj = self.y_train # for measure assessing
        else:
            traj = self.y_pred # for evaluating prediction
        
        
        tree = scipy.spatial.cKDTree(traj)
        self.nn_index = tree.query(traj, k=2)[1]
        
        #print(self.nn_index.shape)
        #drop all elements in nn_index lists where the neighbour is:
        #1. less than 200 timesteps away
        #2. where we cannot calculate the neighbours future in tau_max timesteps:
        
        #contains indices of points and the indices of their nn:
        
        self.nn_index = self.nn_index[np.abs(self.nn_index[:,0] - self.nn_index[:,1]) > threshold]
        #print(self.nn_index.shape)
        self.nn_index = self.nn_index[self.nn_index[:,1] + tau_max < traj.shape[0]]
        self.nn_index = self.nn_index[self.nn_index[:,0] + tau_max < traj.shape[0]]
        #print(self.nn_index.shape)
        #Calculate the largest Lyapunov exponent:
        #for storing the results:
        Sum = []
        #loop over differnt tau, to get a functional dependence:
        for tau in taus:
            #print(tau)
            
            S = []#the summed values for all basis points 
            
            #loop over every point in the trajectory, where we can calclutate
            #the future in tau_max timesteps:
            for point, nn in self.nn_index:
                
                S.append(np.log(np.linalg.norm(traj[point+tau] - traj[nn + tau]))) #add one points average s to S
   
                #since there is no else, we only avg over the points that have
                #points in their epsilon environment
            Sum.append((tau*self.dt, np.array(S).mean()))
        self.Sum = np.array(Sum)
        
        slope = (self.Sum[-1,1] - self.Sum[0,1]) / (self.Sum[-1,0] - self.Sum[0,0])
        if plot:
            plt.title('slope: ' + str(slope))
            plt.plot(self.Sum[:,0], self.Sum[:,1])
            plt.plot(self.Sum[:,0], self.Sum[:,0]*slope)            
            plt.xlabel('time dt*tau[steps]')
            plt.ylabel('log_dist_former_neighbours')
            #plt.plot(self.Sum[:,0], self.Sum[:,0]*slope + self.Sum[0,0])
            plt.show()
            
        t1 = time.time()
        if print_switch:
            print('time: ', t1-t0)
            
        if test_measure:
            self.lyapunov_exponent_nn_test = slope
        else:
            self.lyapunov_exponent_nn = slope
        #return slope
        
    def calc_chi(self):
        traj = self.y_pred
        """
        max_lya: maximal Lyapunov exponent
        C_dim: correlation dimension
        <-->, std(--): expectation value, standard deviation for 1000 original
            trajectories
        chi = max_lya - <max_lya> / std(max_lya) + C_dim - <C_dim> / std(C_dim)
        """
        #exp and std for 100 trajectories a 25k points:
        exp_max_lya = 0.82231800951535061
        std_max_lya = 0.019966305991885182
        exp_C_dim = 2.0307003809606998
        std_C_dim = 0.016888285378633988
        
        self.chi = (self.max_lyapunov_nn(traj) - exp_max_lya )/std_max_lya + \
            (self.correlation_dimension(traj) - exp_C_dim )/std_C_dim
            
        #return self.chi
        
    def autocorrelation_zero_crossing(self, signal):
        """
        returns the timestep at which the signal (1d. array) has the first zero
        crossing.
        used for embedding correctly
        """
        #signal = self.y_train        
        interval = signal.shape[0]
        ac = np.zeros(interval)
        
        for tau in np.arange(interval):
            
            
            for t in np.arange(interval):
                ac[tau] += signal[t] \
                    * signal[(t+tau) % interval]
            if np.sign(ac[0]*ac[tau]) == -1.:
                return tau
                break
    
    def W_out_distr(self):
        """
        Shows a histogram of the fitted parameters of W_out, each output
        dimension in an other color
        """
        f = plt.figure(figsize = (10,10))
        for i in np.arange(self.ydim):
            plt.hist(self.W_out[i], bins=30, alpha=0.5, label='W_out['+str(i)+']')
        plt.legend(fontsize=10)
        f.show()
        '''
        np.argsort(-B.W_out[i]) : arguments of descending elements
        
   
        
        -np.sort(-B.W_out[i])
        shows a decending list of W_out parameters -> for i = 0 only one of
        order 1, for y,z more natural decay
        -> why? (regularization_parameter already decreased)
        '''
    def do_complete_network(self):
        """
        If extended states (not only the reservoir states are used for fitting
        W_out, but also the input x) are used, the adjacency matrix is
        completed to depict this, by appending a unit matrix on the diagonal
        """
        if self.extended_states:
            #larger adjacency matrix is initialized:
            self.complete_network = np.zeros((self.N + self.xdim,
                                          self.N + self.xdim))
            #and filled with the old adj.mat and the unit matrix:
            self.complete_network[:self.N, :self.N] = self.network
            #maybe zero instead of eye is suitable: 
            self.complete_network[self.N: ,self.N:] = np.eye(self.xdim)
        
        else:
            self.complete_network = self.network
                
    def draw_network(self):
        """"
        Shows a plot of the network plus the direct input
        
        want to colorcode the nodes that are most connected to the output
        (through all three dimensions)
        hopefully, one can see some connection to structure in the network
        """        
        self.do_complete_network()
        graph = nx.from_numpy_matrix(self.complete_network)
        
        labels = dict([ (i,i) for i in range(self.complete_network.shape[0])])
        plt.figure(figsize = (12,12))
        nx.draw(graph, nx.circular_layout(graph), node_color = 'r',
                node_size = 30, alpha = 0.5, labels=labels)
        nx.draw_networkx_nodes(graph, nx.circular_layout(graph),
                               node_color='darkgreen', nodelist=self.top_ten,
                               node_size=200, alpha=0.8)
        plt.show()
        """
        since we added eye to the complete_network the in/out_degree counts
        differ from the drawn graph, bc. the connection to a node itelf is not
        displayed but counted!
        """
        
    def network_measures(self):
        """
        Uses the network to calculate the degree distribution in two ways
        """
        '''
        => use in/out degree from adj. matrix instead of nx.degree
        requires W_out from train()
        '''        
        ### in/out_degree
        self.calc_binary_network()
        #self.do_complete_network()
        degree = nx.degree_histogram(nx.from_numpy_matrix(self.network))
        #print(degree)       
        plt.plot(np.arange(len(degree)), degree, 'x')
        plt.title('nx.degree distribution plot of the nextwork')        
        plt.loglog()
        plt.show()
        top_ten = list(np.argsort(-np.sum(self.W_out, axis = 0))[:10])
        print(nx.degree(nx.from_numpy_matrix(self.network), top_ten))
        #nx.to_numpy_matrix: edges from i to j, become entries at A[i,j]
        #-> row_sums (axis=1) are out_degree
        in_degree = self.binary_network.sum(axis=0)
        out_degree = self.binary_network.sum(axis=1)
        plt.hist(in_degree, bins=15, label='in_degree', alpha=0.5, log=True, histtype='stepfilled')
        plt.hist(out_degree, bins=15, label='out_degree', alpha=0.5, log=True, histtype='stepfilled')    
        plt.legend()
        plt.title('in/out_degree distribution plot of the network')
        plt.show()
        #print(in_degree[top_ten], out_degree[top_ten])
        
    def calc_strength(self):
        """
        calculate the absolute in and out strength of nodes and its respective 
        average, by summing over each row of abs(network).
        """
        self.in_strength = np.abs(self.network).sum(axis=0)
        self.avg_in_strength = self.in_strength.mean()
        self.out_strength = np.abs(self.network).sum(axis=1)
        self.avg_out_strength = self.out_strength.mean()
        
    def calc_clustering_coeff(self):
        """
        clustering coefficient for each node and avg
        """        
        network = self.binary_network
        k = network.sum(axis=0)
        C = np.diag(np.matmul(np.matmul(network, network), network))/ k*(k-1)   
        self.clustering_coeff = C
        self.avg_clustering_coeff = C.mean()
        
    def plt(self, flag='y', save_switch=False, path=''):
        """
        Shows timeresolved plots of:
        'r': some of the reservoir states
        'y': the predicted and true future states
        't': the x,y plot of the trajectory
        """
        ### plotting parameter in dict
        train_dict = { 'color': 'green', 'alpha': 0.2, 'label': 'train'}
        pred_dict = { 'color': 'red', 'alpha': 0.4, 'label': 'pred'}
            
        if flag == 't':
            
            f = plt.figure(figsize = (15,15))
            ax1 = f.add_subplot(111)
            
            ax1.set_title('x,y plot of prediction')
            ax1.plot(self.y_pred[:,0], self.y_pred[:,1], **pred_dict)
            ax1.plot(self.y_test[:,0], self.y_test[:,1], **train_dict)
            ax1.legend()
            
        elif flag == 'r':
            print('take care which split was used in calc_top_ten')
            split_calc = self.top_ten_bool_1d.sum()
            f = plt.figure(figsize = (15,15))
            ax1 = f.add_subplot(121)
            ax1.set_ylim((-1.05, 1.05))
            ax1.set_title('network during training')
            ax1.plot(np.arange(self.r.shape[0]),
                     np.reshape(self.r[:,self.top_ten_bool_1d],
                                (self.r.shape[0], split_calc)))
                
            ax2 = f.add_subplot(122)
            ax2.set_ylim((-1.05, 1.05))        
            ax2.set_title('network during prediction')
            ax2.plot(np.arange(self.r_pred.shape[0]),
                     np.reshape(self.r_pred[:,self.top_ten_bool_1d],
                                (self.r.shape[0], split_calc)))

        elif flag == 'y':
            
            f = plt.figure(figsize = (15,15))
            
            ### x-range
            time_range_pred = np.arange(
                self.training_steps,
                self.training_steps + self.prediction_steps)
            time_range_train = np.arange(self.training_steps)
            
            
            #y-values
            if self.system == lorenz.roessler:
                ylimits = ((-5,8), (-8,5), (0, 15))
            elif self.system == lorenz.mod_lorenz:
                ylimits = ((-15,20), (-15,30), (-10,50))
            else:
                ylimits = ((self.y_test[:,0].min(), self.y_test[:,0].max()),
                           (self.y_test[:,1].min(), self.y_test[:,1].max()),
                            (self.y_test[:,2].min(), self.y_test[:,2].max()))
            for i, ylim in enumerate(ylimits):
                ax1 = plt.subplot(3,2,2*i+1)
                ax1.set_ylim(ylim)
                ax1.set_title('y['+str(i)+']value_train')
                ax1.plot(time_range_train, self.y_train[:,i], **train_dict)
                
                ax2 = plt.subplot(3,2,2*i+2)
                ax2.set_title('y['+str(i)+']_value_pred')
                ax2.set_ylim(ylim)
                ax2.plot(time_range_pred[:], self.y_pred[:,i], **pred_dict)
                ax2.plot(time_range_pred[:], self.y_test[:,i], **train_dict)
                ax2.legend()

        if save_switch:
            print(path)
            if path:
                f.savefig(filename=str(path), format='pdf')
            else:
                print('path argument need in self.plt()')                
        else:
            print('normal show')
            f.show()
            
    def exp_dev(self):
        """
        calculate the expectation value and std deviation for i trajectories
        with length 25000 of lyapunov exp and correlation dim for y_train
        sum(axis=0).mean()
        """
        i = 100
        self.lya = np.zeros(i)
        self.C_dim = np.zeros(i)
        for j in np.arange(0, 25000*i, 25000):
            print(j/25000, 'of 100')
            traj = self.y_train[j:j+25000]
            self.lya[j/25000] = self.max_lyapunov_nn(traj)
            self.C_dim[j/25000] = self.correlation_dimension(traj)
            print(traj[0,0])
            #self.load_data()
            
    def calc_weighted_clustering_coeff_onnela(self):
        """
        weigthed clustering coef from Onnela2005 paper
        using absolute values of normalized network weights as basis.
        Replace NaN (originating from division by zero (degree = 0,1)) with 0.
        """
        k = self.binary_network.sum(axis=0)
        #print(k)
        network = abs(self.network)/ abs(self.network).max()
        
        self.weighted_cc = np.diag(np.matmul(np.cbrt(network),
                                 np.matmul(np.cbrt(network), np.cbrt(network)))) \
               / (k*(k-1))
        # assign 0. to infinit values:
        self.weighted_cc[np.isnan(self.weighted_cc)] = 0.
               
        self.avg_weighted_cc = self.weighted_cc.mean()      
                

            
            
            
            
          
#if __main__:

### INLINE COMMANDS
#==============================================================================
#A = reservoir(number_of_nodes=500, input_dimension=3,
#                 output_dimension=3, type_of_network='random',
#                 dt = 2e-2, training_steps=5000,
#                 prediction_steps=5000, discard_steps=5000,
#                 regularization_parameter=0.0001, spectral_radius=1.5,
#                 input_weight=.7, avg_degree=6., epsilon=np.array([5,10,5]),
#                 extended_states=False)
#A.vary_network()
#A.load_data()
#A.train()
#A.predict()
#A.RMSE('pred')
#A.plt('t')

