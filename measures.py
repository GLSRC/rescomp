# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:41:00 2019

@author: aumeier, edited slightly by baur
"""
#from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal
import scipy.sparse.linalg
import scipy.spatial

import sys
sys.path.append("..")
sys.path.append(".")
try: import esn_rescomp
except ModuleNotFoundError: from . import esn_rescomp


# import sys
# sys.path.append(".")
# from . import esn_rescomp


# from . import esn_rescomp


class reservoir(esn_rescomp.res_core):
    def __init__(self, **kwargs):
        self.demerge_time = None  # demerge_time() assigns value
        self.lyapunov = None  # max_lyapunov_nn() assigns value
        self.lyapunov_test = None
        self.dimension = None  # correlation_dimension(traj) assigns value
        self.dimension_test = None
        # self.chi = None #calc_chi
        self.in_strength = None
        self.avg_in_strength = None
        self.out_strength = None
        self.avg_out_strength = None
        self.avg_weighted_cc = None
        self.weighted_cc = None
        self.clustering_coeff = None
        self.avg_clustering_coeff = None
        super(reservoir, self).__init__(**kwargs)

    def RMSE(self, flag):
        """
        Measures an average over the spatial distance of predicted and actual
        trajectory
        """
        if flag == 'train':
            error = np.sqrt(((np.matmul(self.W_out, self.r.T) - \
                              self.y_train.T) ** 2).sum() \
                            / (self.y_train ** 2).sum())
            self.RMSE_train = error
        elif flag == 'pred':
            error = np.sqrt(((np.matmul(self.W_out, self.r_pred.T) - \
                              self.y_test.T) ** 2).sum() \
                            / (self.y_test ** 2).sum())
            self.RMSE_pred = error
        else:
            raise Exception('use "train" or "pred" as flag')

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

    def calc_dimension(self, r_min=0.5, r_max=5., r_steps=0.15,
                       plot=False, test_measure=False):
        """
        Calculates correlation dimension for self.y_pred (or self.y_test) using
        the algorithm by Grassberger and Procaccia and stores it in
        self.dimension.
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
            traj = self.y_test  # for measure assessing
        else:
            traj = self.y_pred  # for evaluating prediction

        nr_points = float(traj.shape[0])
        radii = np.arange(r_min, r_max, r_steps)

        tree = scipy.spatial.cKDTree(traj)
        N_r = np.array(tree.count_neighbors(tree, radii), dtype=float) / nr_points
        N_r = np.vstack((radii, N_r))

        # linear fit based on loglog scale, to get slope/dimension:
        slope, intercept = np.polyfit(np.log(N_r[0]), np.log(N_r[1]), deg=1)[0:2]
        if test_measure:
            self.dimension_test = slope
        else:
            self.dimension = slope

        ###plotting
        if plot:
            plt.loglog(N_r[0], N_r[1], 'x', basex=10., basey=10.)
            plt.title('loglog plot of the N_r(radius), slope/dim = ' + str(slope))
            plt.show()

    def calc_lyapunov(self, threshold=int(10),
                      plot=False, print_switch=False, test_measure=False):
        """
        Calculates the maximal Lyapunov Exponent of self.y_pred (or self.y_test),
        by estimating the time derivative of the mean logarithmic distances of
        former next neighbours. Stores it in self.lyapunov (self.lyapunov_test)
        Only values for tau_min/max are used for calculating the slope!
        
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
        tau_min = int(0.5 / self.dt)
        tau_max = int(3.8 / self.dt)
        taus = np.arange(tau_min, tau_max, 10)  # taus = np.array([tau_min, tau_max])

        if test_measure:
            traj = self.y_test  # for measure assessing
        else:
            traj = self.y_pred  # for evaluating prediction

        tree = scipy.spatial.cKDTree(traj)
        self.nn_index = tree.query(traj, k=2)[1]
        
        #drop all elements in nn_index lists where the neighbour is:
        #1. less than threshold timesteps away
        #2. where we cannot calculate the neighbours future in tau_max timesteps:
        
        #contains indices of points and the indices of their nn:
        self.nn_index = self.nn_index[np.abs(self.nn_index[:, 0] - self.nn_index[:, 1]) > threshold]

        self.nn_index = self.nn_index[self.nn_index[:, 1] + tau_max < traj.shape[0]]
        self.nn_index = self.nn_index[self.nn_index[:, 0] + tau_max < traj.shape[0]]

        # Calculate the largest Lyapunov exponent:
        # for storing the results:
        Sum = []
        # loop over differnt tau, to get a functional dependence:
        for tau in taus:
            # print(tau)

            S = []  # the summed values for all basis points

            # loop over every point in the trajectory, where we can calclutate
            # the future in tau_max timesteps:
            for point, nn in self.nn_index:
                S.append(np.log(np.linalg.norm(traj[point + tau] - traj[nn + tau])))  # add one points average s to S

                # since there is no else, we only avg over the points that have
                # points in their epsilon environment
            Sum.append((tau * self.dt, np.array(S).mean()))
        self.Sum = np.array(Sum)

        slope = (self.Sum[-1, 1] - self.Sum[0, 1]) / (self.Sum[-1, 0] - self.Sum[0, 0])
        if plot:
            plt.title('slope: ' + str(slope))
            plt.plot(self.Sum[:, 0], self.Sum[:, 1])
            plt.plot(self.Sum[:, 0], self.Sum[:, 0] * slope)
            plt.xlabel('time dt*tau[steps]')
            plt.ylabel('log_dist_former_neighbours')
            # plt.plot(self.Sum[:,0], self.Sum[:,0]*slope + self.Sum[0,0])
            plt.show()

        if test_measure:
            self.lyapunov_test = slope
        else:
            self.lyapunov = slope

    def W_out_distr(self):
        """
        Shows a histogram of the fitted parameters of self.W_out, each output
        dimension in an other color
        """
        f = plt.figure(figsize=(10, 10))
        for i in np.arange(self.ydim):
            plt.hist(self.W_out[i], bins=30, alpha=0.5, label='W_out[' + str(i) + ']')
        plt.legend(fontsize=10)
        f.show()

    def calc_strength(self):
        """
        Calculate the absolute in and out strength of nodes in self.network
        and its respective average.
        Stores them in :self.in_strength, self.avg_in_strength, self.out_strength, 
        self.avg_out_strength
        """
        self.in_strength = np.abs(self.network).sum(axis=0)
        self.avg_in_strength = self.in_strength.mean()
        self.out_strength = np.abs(self.network).sum(axis=1)
        self.avg_out_strength = self.out_strength.mean()

    def calc_clustering_coeff(self):
        """
        clustering coefficient for each node and avg, and stores them in
        self.clustering_coeff and self.avg_clustering_coeff.
        """
        network = self.binary_network
        k = network.sum(axis=0)
        C = np.diag(np.matmul(np.matmul(network, network), network)) / k * (k - 1)
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
        train_dict = {'color': 'green', 'alpha': 0.2, 'label': 'train'}
        pred_dict = {'color': 'red', 'alpha': 0.4, 'label': 'pred'}

        if flag == 't':
            f = plt.figure(figsize=(15, 15))
            ax1 = f.add_subplot(111)

            ax1.set_title('x,y plot of prediction')
            ax1.plot(self.y_pred[:, 0], self.y_pred[:, 1], **pred_dict)
            ax1.plot(self.y_test[:, 0], self.y_test[:, 1], **train_dict)
            ax1.legend()

        elif flag == 'r':
            split = 0.1
            f = plt.figure(figsize=(15, 15))
            ax1 = f.add_subplot(121)
            ax1.set_ylim((-1.05, 1.05))
            ax1.set_title('network during training')
            ax1.plot(np.arange(self.r.shape[0]),
                     self.r[:, self.calc_tt(flag='bool_1d', split=split)])

            ax2 = f.add_subplot(122)
            ax2.set_ylim((-1.05, 1.05))
            ax2.set_title('network during prediction')
            ax2.plot(np.arange(self.r_pred.shape[0]),
                     self.r_pred[:, self.calc_tt(flag='bool_1d', split=split)])

        elif flag == 'y':
            # f = plt.figure(figsize = (15,15))
            f, axes = plt.subplots(self.ydim, 2, figsize=(15, 15))

            ### x-range
            time_range_pred = np.arange(
                self.training_steps,
                self.training_steps + self.prediction_steps)
            time_range_train = np.arange(self.training_steps)

            # y-values
            if self.sys_flag == "roessler":
                ylimits = ((-5, 8), (-8, 5), (0, 15))
            elif self.sys_flag == 'mod_lorenz':
                ylimits = ((-25, 20), (-30, 30), (0, 50))
            else:
                traj = np.concatenate((self.y_test, self.y_train), axis=0)
                ylimits = [[traj[:, dim].min(), traj[:, dim].max()] for dim in np.arange(self.ydim)]
            for i, ylim in enumerate(ylimits):
                ax1 = axes[i][0]
                ax1.set_ylim(ylim)
                ax1.set_title('y[' + str(i) + ']_value_train')
                ax1.plot(time_range_train, self.y_train[:, i], **train_dict)

                # ax2 = plt.subplot(self.ydim,2,2*i+2)
                ax2 = axes[i][1]
                ax2.set_title('y[' + str(i) + ']_value_pred')
                ax2.set_ylim(ylim)
                ax2.plot(time_range_pred[:], self.y_pred[:, i], **pred_dict)
                ax2.plot(time_range_pred[:], self.y_test[:, i], **train_dict)
                ax2.legend()

        else:
            raise Exception("Plot flag couldn't be resolved")

        if save_switch:
            print(path)
            if path:
                f.savefig(filename=str(path), format='pdf')
            else:
                raise Exception('path argument needed in self.plt()')
        else:
            print('normal show')
            f.show()

    def calc_weighted_clustering_coeff_onnela(self):
        """
        Calculates the weighted clustering coefficient of abs(self.network)
        according to Onnela paper from 2005.
        Replacing NaN (originating from division by zero (degree = 0,1)) with 0.
        Stores it in self.weighted_cc and its mean in self.avg_weighted_cc.
        """
        k = self.binary_network.sum(axis=0)
        # print(k)
        network = abs(self.network) / abs(self.network).max()

        self.weighted_cc = np.diag(np.matmul(np.cbrt(network),
                            np.matmul(np.cbrt(network), np.cbrt(network)))) \
                           / (k * (k - 1))
        # assign 0. to infinit values:
        self.weighted_cc[np.isnan(self.weighted_cc)] = 0.

        self.avg_weighted_cc = self.weighted_cc.mean()

#    def calc_covar_rank(self, flag='train'):
#        """
#        Calculated the covarianc rank of the squared network dynamics matrix self.r
#        (or self.r_pred) and stores it in self.covar_rank               
#        """
#        """
#        Does not calculate the actual covariance matrix!! Fix befor using
#        """
#        if flag == 'train':
#            res_dyn = self.r
#        elif flag == 'pred':
#            res_dyn = self.r_pred
#        else:
#            raise Exception("wrong covariance flag")
#        covar = np.matmul(res_dyn.T, res_dyn)
#        #self.covar_rank = np.linalg.matrix_rank(covar)
#        print(np.linalg.matrix_rank(covar))
        
    def remove_nodes(self, split):
        """
        This method removes nodes from the network and W_in according to split,
        updates avg_degree, spectral_radius, 
        This new reservoir is returned
        split should be given as a list of two values or a float e [-1. and 1.]
        example: split = [-.3, 0.3]
        """
        if type(split) == list:
            if len(split) < 3:
                pass
            else:
                raise Exception('too many entries in split. length: ', len(split))
        elif type(split) == float and split >= -1. and split <= 1.:
            split = [split]
        else:
            raise Exception('values in split not between -1. and 1., type: ', type(split))

        remaining_size = sum(np.abs(split))
        new = reservoir(sys_flag=self.sys_flag, N=int(round(self.N * (1 - remaining_size))),
                        input_dimension=3, output_dimension=3,
                        type_of_network=self.type, dt=self.dt,
                        training_steps=self.training_steps,
                        prediction_steps=self.prediction_steps,
                        discard_steps=self.discard_steps,
                        regularization_parameter=self.reg_param,
                        spectral_radius=self.spectral_radius,
                        input_weight=self.input_weight, avg_degree=self.avg_degree,
                        epsilon=self.epsilon, extended_states=self.extended_states,
                        W_in_sparse=self.W_in_sparse, W_in_scale=self.W_in_scale)
        # gather to be removed nodes arguments in rm_args:
        rm_args = np.empty(0)
        for s in split:
            rm_args = np.append(self.calc_tt(flag='arg', split=s), rm_args)
            # print(s, rm_args.shape)

        # rows and columns of network are deleted according to rm_args:
        new.network = np.delete(np.delete(self.network, rm_args, 0), rm_args, 1)

        # the new average degree is calculated:
        new.calc_binary_network()
        new.avg_degree = new.binary_network.sum(axis=0).mean(axis=0)
        # the new spectral radius is calculated:
        new.network = scipy.sparse.csr_matrix(new.network)
        try:
            eigenvals = scipy.sparse.linalg.eigs(new.network, k=1, which='LM')[0]
            new.spectral_radius = np.absolute(eigenvals).max()
            # try:
            #     eigenvals = scipy.sparse.linalg.eigs(new.network, k=1, which='LM')[0]
            #     new.spectral_radius = np.absolute(eigenvals).max()
            # except:
            #     print('eigenvalue calculation failed!, no spectral_radius assigned')

            new.network = new.network.toarray()

        except ArpackNoConvergence as conv_error:
            print(conv_error)
            
        
        # Adjust W_in
        new.W_in = np.delete(self.W_in, rm_args, 0)
        # pass x,y to new_reservoir
        new.x_train = self.x_train
        new.x_discard = self.x_discard
        new.y_test = self.y_test
        new.y_train = self.y_train
        
        return new
