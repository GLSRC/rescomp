from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy as sc
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import attractorAnalysis_CD as aa
import pickle

" Setup and parameters "
trainLen = 5000 # T/delta_t
testLen = 9999
initLen = 100
inSize = outSize = 3
resSize = 300
a = 1.0 # leaking rate, in pathak paper a = 1.0
reg = 0.01  # 1e-8# regularization coefficient (discourages overfitting by penalizing large values), beta = 0 leads to problems when inverting X*X_T matrix

#### Select network #####
nw_type = "erdos_renyi" # small_world, erdos_renyi, scale_free

# load data
data = np.loadtxt('lorenz_mod.txt')

# Set statistical dimensionality
plot_results = "no" # yes
seedNumber = 1000
rhoTargetRange = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.8, 2.0, 2.2, 2.4])

# Variables for storage
fcHorizon = np.zeros((len(rhoTargetRange),seedNumber))
corrDimAll  = np.zeros((len(rhoTargetRange),seedNumber))
lyapMaxAll  = np.zeros((len(rhoTargetRange),seedNumber))

for j in range(len(rhoTargetRange)):
    rhoTarget = rhoTargetRange[j]
    print("j: ", j)
    for i in range(1,seedNumber+1):

        print(i)
        seed2 = i 
        np.random.seed(seed2)

        p = 0.0# dummy
        if nw_type == "small_world":
            p = 0.2 #0.015 # probability of rewiring each edge
            k = 7#10 # Each node is connected to k nearest neighbors in ring topology. 7 leads to roughly same number of edges as for Erdos Renyi setup
            network = nx.watts_strogatz_graph(resSize, k, p, seed=seed2)
            
        elif nw_type == "connected_small_world":        
            p = 0.25 # probability of rewiring each edge
            k = 10 # Each node is connected to k nearest neighbors in ring topology
            network = nx.connected_watts_strogatz_graph(resSize, k, p, seed=seed2)
            
        elif nw_type == "erdos_renyi":
            p = 0.02 # default 0.02

            network = nx.gnp_random_graph(resSize, p , seed=seed2) # create Erdös Renyi network (gnp graph) with d = n*p
            
        elif nw_type == "scale_free":
            network = nx.scale_free_graph(resSize, alpha=0.285, beta=0.665, gamma=0.05, delta_in=0.2, delta_out=0, create_using=None, seed=seed2) # see networkx for defs for alpha, beta, gamma. Sum over the three must be 1
            #network = nx.scale_free_graph(resSize, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0,  create_using=None, seed=seed2)

        W = np.array(nx.to_numpy_matrix(network))
        numNonzero = np.count_nonzero(W)
        indsNonzero = np.nonzero(W)
        elementsNonzero = np.random.uniform(-1,1,numNonzero) # nonzero elements of adjancency matrix are drawn uniformly
        W[indsNonzero] = elementsNonzero # replace nonzero elements with above generated random numbers
        
        # Normalize reservoir such that adjacency matrix has desired spectral radius
        rhoW = max(abs(sc.linalg.eig(W)[0]))
        W *= rhoTarget / rhoW
        
        # Scale input signal based on distribution of nonzero elements of the reservoir
        inputScale = 1*(rhoTarget / rhoW)
        Win = np.random.uniform(-inputScale,inputScale,(resSize,inSize))
        
        # setup the design (collected states) matrix
        X = np.zeros((resSize,trainLen-initLen))
        
        # target matrix - here trajectory of the lorenz system
        Yt = data[:,initLen+1:trainLen+1]
        
        # run the reservoir with the data and collect X
        x = np.zeros((resSize,1))
        for t in range(trainLen):
            u = data[:,t]
            u = np.reshape(u,(np.size(u),1))
            x = (1-a)*x + a*np.tanh( np.dot( Win, u ) + np.dot( W, x ) )
            if t >= initLen:
                X[:,t-initLen] = x[:,0]

        # train the output using ridge regression / Tikhonov regularization
        Wout = np.dot(np.dot(Yt, X.T), sc.linalg.inv(np.dot(X, X.T) + \
                                            reg * np.eye(resSize)))
        
        # run the trained ESN in prediction mode
        Y = np.zeros((outSize, testLen))
        u = data[:,trainLen]
        u = np.reshape(u,(np.size(u),1))
        Xp = np.zeros((resSize, testLen))
        for t in range(testLen):
            x = (1 - a) * x + a * np.tanh(np.dot(Win, u) + np.dot(W, x))
            y = np.dot(Wout, x)
            Y[:, t] = y[:,0] # I guess that is not very pythonic ..
            Xp[:,t-initLen] = x[:,0] # Store reservoir states
            u = y


        " Plot results "
        if plot_results == "yes":
            sns.set(font_scale=1.1)
            sns.set_style("ticks", {"xtick.direction": u"in", "ytick.direction": u"in"})
            f, axarr = plt.subplots(3, 1, figsize=(6, 5.5))
            
            axarr[0].plot(data[0,trainLen + 1:trainLen + testLen + 1], 'g')
            axarr[0].plot(Y.T[:,0], 'b')
            axarr[0].set_ylabel('X')
            
            axarr[1].plot(data[1,trainLen + 1:trainLen + testLen + 1], 'g')
            axarr[1].plot(Y.T[:,1], 'b')
            axarr[1].set_ylabel('Y')
            
            axarr[2].plot(data[2,trainLen + 1:trainLen + testLen + 1], 'g')
            axarr[2].plot(Y.T[:,2], 'b')
            axarr[2].set_ylabel('Z')
            
            f.tight_layout()
            #plt.show()
            #plt.savefig('ZZZ_Prediction_1DInput_5th_t-1_'+str(i)+nw_type+'.png', dpi=150, format='png')
            
            directory = 'Plots/Rho_'+str(rhoTarget)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            f.savefig('Plots/Rho_'+str(rhoTarget)+'/ESN_10000_Prediction_'+str(i)+nw_type+'p_'+str(p)+'.png', dpi=100, format='png')
            plt.close('all')

        " Forecast Horizon "
        # We are interest in the time during which the forecast stays close to the original series "
        outputDiff = Y - data[:, trainLen + 1:trainLen + testLen + 1]
        thresholdDiffX = 2.5  # threshold for output quality calc
        thresholdDiffZ = 4
        fc_x = np.argmax(np.abs(outputDiff[0, :]) > thresholdDiffX)
        fc_y = np.argmax(np.abs(outputDiff[1, :]) > thresholdDiffX)
        fc_z = np.argmax(np.abs(outputDiff[2, :]) > thresholdDiffZ)

        fc_all = np.array([fc_x, fc_y, fc_z])
        fc_all[fc_all == 0] = testLen
        forecastHorizon = np.min(
            fc_all)  # acts on flatenned array, thus reflects whatever dimension hits threshold first

        " Characterize the long term climate of the attractor"
        try:
           corrDim = aa.corr_dim(Y, rvals=None, dist="rowwise_euclidean", fit="RANSAC", debug_plot=False, debug_data=False, plot_file=None)
        except:
           corrDim = -999

        timeGrid = 1/0.02 #
        try:
           lyapMax = aa.lyap_r2(Y, min_tsep=150, trajectory_len=170)*timeGrid #check
        except:
           lyapMax = -999

        " Append Relevant Measures "
        fcHorizon[j,i-1] = forecastHorizon
        corrDimAll[j,i-1] = corrDim
        lyapMaxAll[j,i-1] = lyapMax


directory = 'Output'
if not os.path.exists(directory):
    os.makedirs(directory)

" Save Measures and Measures "
f = open('Output/LorenzMod_Corr_Dim_10000_'+nw_type+'p_'+str(p)+'.pckl', 'wb')
pickle.dump(corrDimAll, f)
f.close()

f = open('Output/LorenzMod_Max_Lyap_10000_'+nw_type+'p_'+str(p)+'.pckl', 'wb')
pickle.dump(lyapMaxAll, f)
f.close()

f = open('Output/LorenzMod_FC_Hor_10000'+nw_type+'p_'+str(p)+'.pckl', 'wb')
pickle.dump(fcHorizon, f)
f.close()




