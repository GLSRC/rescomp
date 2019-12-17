# -*- coding: utf-8 -*-
""" Reservoir Computing example for the higher dimensional KS system

@author: baur
"""
import numpy as np
import matplotlib.pyplot as plt
from rescomp import esn
from rescomp.simulations import ks_pde_simulation

if __name__ == "__main__":

    # Create 100 dimensional input by simulating the Kuramoto–Sivashinsky PDE
    data = ks_pde_simulation(dimensions=100, system_size=25, t_max=500,
                             time_step=0.05)

    # Used to always create the same random network and hence prediction
    np.random.seed(0)

    # For systems other than the Lorenz63 system used in the minimal_example,
    # further hyperparameters have to be varied/optimized. Especially important
    # are the regularization, the spectral radius and the average network degree
    # If you want to see the effect of badly chosen hyperparameters, set e.g.
    # the spectral radius to 0.9 or, for a completely failed prediction, set it
    # to 1.6 or higher
    esn = esn(network_dimension=5000, input_dimension=data.shape[1],
              output_dimension=data.shape[1], training_steps=6000,
              prediction_steps=500, discard_steps=999,
              regularization_parameter=0.01, spectral_radius=0.3,
              avg_degree=100)

    esn.load_data(data)
    esn.train()
    esn.predict()

    # --- Plot the results

    pred = esn.y_pred
    test = esn.y_test

    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(3, 1, 1)
    ax1.imshow(test.T)
    ax1.set_title("Simulation")

    ax2 = fig1.add_subplot(3, 1, 2)
    ax2.imshow(pred.T)
    ax2.set_title("Prediction")

    ax3 = fig1.add_subplot(3, 1, 3)
    ax3.imshow(pred.T - test.T)
    ax3.set_title("Difference between simulation and prediction")

    plt.show()

