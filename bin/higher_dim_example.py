# -*- coding: utf-8 -*-
""" Reservoir Computing example for the higher dimensional KS system

@author: baur
"""
import numpy as np
import matplotlib.pyplot as plt
import rescomp
# from rescomp import ESNOld
from rescomp.simulations import kuramoto_sivashinsky

if __name__ == "__main__":

    # Setup data length variables
    discard_steps = 1000
    training_steps = 50000
    prediction_steps = 500
    total_steps = discard_steps + training_steps + prediction_steps + 1

    print("Start the KS simulation")
    # Create 100 dimensional input by simulating the Kuramoto–Sivashinsky PDE
    data = kuramoto_sivashinsky(dimensions=100, system_size=22, dt=0.05,
                                time_steps=total_steps)
    print("KS Simulation done")

    # Used to always create the same random network and hence prediction
    np.random.seed(0)

    # For systems other than the Lorenz63 system used in the minimal_example,
    # further hyperparameters have to be varied/optimized. Especially important
    # are the regularization, the spectral radius and the average network degree
    # If you want to see the effect of badly chosen hyperparameters, set e.g.
    # the spectral radius to 0.9 or, for a completely failed prediction, set it
    # to 1.6 or higher
    esn = rescomp.esn.ESNOld(network_dimension=5000, input_dimension=data.shape[1],
              output_dimension=data.shape[1], training_steps=training_steps,
              prediction_steps=prediction_steps, discard_steps=discard_steps,
              regularization_parameter=0.01, spectral_radius=0.3,
              avg_degree=100)

    print("Start Reservoir Computing")

    esn.load_data(data)
    esn.train()
    esn.predict()

    print("Reservoir Computing Done")

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

