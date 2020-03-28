# -*- coding: utf-8 -*-
""" Test if the resmop.measures module works as it should
"""

import unittest
import numpy as np
import rescomp
from rescomp import measures

class test_measures(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        np.random.seed(None)

    def test_rmse(self):
        length = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))

        norm = pred.shape[0]
        rmse_desired = np.sqrt(((pred - meas) ** 2).sum() / norm)

        rmse = measures.rmse(pred, meas)

        # results not exactly equal due to numpy optimizations
        np.testing.assert_allclose(rmse, rmse_desired, rtol=1e-15)

    def test_nrmse(self):
        length = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))

        norm = (meas ** 2).sum()
        nrmse_desired = np.sqrt(((pred - meas) ** 2).sum() / norm)

        nrmse = measures.nrmse(pred, meas)

        # results not exactly equal due to numpy optimizations
        np.testing.assert_allclose(nrmse, nrmse_desired, rtol=1e-15)

    def test_nrmse_over_time(self):
        length = 1
        dim = np.random.randint(10, 100)

        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))

        nrmse_desired = measures.nrmse(pred, meas)
        nrmse = measures.nrmse_over_time(pred, meas)

        np.testing.assert_equal(nrmse, nrmse_desired)

    def test_divergence_time(self):
        pred = np.array([[i, i+1] for i in range(10)])
        meas = np.array([[i*2, i+1] for i in range(10)])

        epsilon = 5

        div_time_desired = 6
        div_time = measures.divergence_time(pred, meas, epsilon)

        np.testing.assert_equal(div_time, div_time_desired)


if __name__ == "__main__":
    unittest.main(verbosity=2)
