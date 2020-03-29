# -*- coding: utf-8 -*-
""" Tests if the rescomp.utilities module works as it should """

import unittest
import rescomp
import numpy as np

class test_utilities(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_train_and_predict_input_setup(self):
        disc_steps = 3
        train_sync_steps = 2
        train_steps = 4
        pred_steps = 5
        total_time_steps = disc_steps + train_sync_steps + train_steps + \
                           pred_steps

        x_dim = 3
        data = np.random.random((total_time_steps, x_dim))

        x_train_desired = data[disc_steps: disc_steps + train_sync_steps + train_steps]
        x_pred_desired = data[disc_steps + train_sync_steps + train_steps - 1:]

        x_train, x_pred = rescomp.utilities.train_and_predict_input_setup(
            data, disc_steps=disc_steps, train_sync_steps=train_sync_steps,
            train_steps=train_steps, pred_steps=pred_steps)

        np.testing.assert_equal(x_train, x_train_desired)
        np.testing.assert_equal(x_pred, x_pred_desired)



if __name__ == "__main__":
    unittest.main(verbosity=2)
