# -*- coding: utf-8 -*-
""" Tests if the rescomp.measures module works as it should """

import unittest
import numpy as np
from rescomp import measures

class testMeasures(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        np.random.seed(None)

    def test_rmse(self):
        length = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))

        rmse_desired = np.sqrt(((pred - meas) ** 2).sum() / meas.shape[0])

        rmse = measures.rmse(pred, meas)

        # results not exactly equal due to numpy optimizations
        np.testing.assert_allclose(rmse, rmse_desired, rtol=1e-15)

    def test_rmse_normalization_mean(self):
        length = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))

        nrmse_desired = \
            np.sqrt(((pred - meas) ** 2).sum() / meas.shape[0]) / np.mean(meas)

        nrmse = measures.rmse(pred, meas, normalization="mean")

        # results not exactly equal due to numpy optimizations
        np.testing.assert_allclose(nrmse, nrmse_desired, rtol=1e-15)

    def test_rmse_normalization_std_over_time(self):
        length = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))

        std = np.std(meas, axis=0)
        mean_std = np.mean(std)
        nrmse_desired = \
            np.sqrt(((pred - meas) ** 2).sum() / meas.shape[0]) / mean_std

        nrmse = measures.rmse(pred, meas, normalization="std_over_time")

        # results not exactly equal due to numpy optimizations
        np.testing.assert_allclose(nrmse, nrmse_desired, rtol=1e-15)

    def test_rmse_over_time(self):
        length = 1
        dim = np.random.randint(10, 100)

        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))

        rmse_desired = measures.rmse(pred, meas)
        rmse = measures.rmse_over_time(pred, meas)

        np.testing.assert_equal(rmse, rmse_desired)

    def test_divergence_time(self):
        pred = np.array([[i, i+1] for i in range(10)])
        meas = np.array([[i*2, i+1] for i in range(10)])

        epsilon = 5

        div_time_desired = 6
        div_time = measures.divergence_time(pred, meas, epsilon)

        np.testing.assert_equal(div_time, div_time_desired)


if __name__ == "__main__":
    unittest.main(verbosity=2)
