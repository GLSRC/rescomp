# -*- coding: utf-8 -*-
""" Tests if the rescomp.utilities module works as it should """

import unittest
from rescomp import utilities
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

        x_train, x_pred = utilities.train_and_predict_input_setup(
            data, disc_steps=disc_steps, train_sync_steps=train_sync_steps,
            train_steps=train_steps, pred_steps=pred_steps)

        np.testing.assert_equal(x_train, x_train_desired)
        np.testing.assert_equal(x_pred, x_pred_desired)

    def test_find_nth_substring(self):
        test_str = '0.s0.0fd.0sf.5'

        self.assertEqual(utilities._find_nth_substring(test_str, '.', 0), None)
        self.assertEqual(utilities._find_nth_substring(test_str, '.', 1), 1)
        self.assertEqual(utilities._find_nth_substring(test_str, '.', 2), 4)
        self.assertEqual(utilities._find_nth_substring(test_str, '.', 3), 8)
        self.assertEqual(utilities._find_nth_substring(test_str, '.', 4), 12)
        self.assertEqual(utilities._find_nth_substring(test_str, '.', 5), None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
