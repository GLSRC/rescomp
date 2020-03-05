# -*- coding: utf-8 -*-
""" Test if the unittest module works as it should

@author: baur
"""

import unittest
import rescomp
import numpy as np


class test_ESN(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.esn = rescomp.ESN()
        self.esn.set_console_logger("off")

    def tearDown(self):
        del self.esn
        np.random.seed(None)

    # TODO: Tests should be much less broad than this, but I am lazy
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

if __name__ == "__main__":
    unittest.main(verbosity=2)
