# -*- coding: utf-8 -*-
""" Tests if the rescomp.measures module works as it should """

import unittest
import numpy as np
from rescomp import measures
import pytest


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
        pred = np.array([[i, i + 1] for i in range(10)])
        meas = np.array([[i * 2, i + 1] for i in range(10)])

        epsilon = 5

        div_time_desired = 6
        div_time = measures.divergence_time(pred, meas, epsilon)

        np.testing.assert_equal(div_time, div_time_desired)

    @pytest.mark.skip(reason='measures.error_over_time not yet implemented')
    def test_error_over_time_same_as_rmse_over_time(self):
        length = np.random.randint(1, 100)
        dim = np.random.randint(1, 100)
        # some norm
        norm = "maxmin"
        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))
        rmse = measures.rmse_over_time(pred, meas, normalization=norm)
        error = measures.error_over_time(pred, meas, distance_measure="rmse", normalization=norm)
        np.testing.assert_almost_equal(rmse, error, decimal=15)

    @pytest.mark.skip(reason='measures.error_over_time not yet implemented')
    def test_error_over_time_custom_function(self):
        length = np.random.randint(1, 100)
        dim = np.random.randint(1, 100)
        # some norm
        norm = None
        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))
        error_str_distance_measure = measures.error_over_time(pred, meas, distance_measure="L2", normalization=norm)

        def L2_function(delta):
            return np.linalg.norm(delta, axis=1)

        error_fct_distance_measure = measures.error_over_time(pred, meas, distance_measure=L2_function,
                                                              normalization=norm)
        np.testing.assert_almost_equal(error_str_distance_measure, error_fct_distance_measure, decimal=15)

    @pytest.mark.skip(reason='measures.error_over_time not yet implemented')
    def test_error_over_time_special_norm(self):
        length = np.random.randint(1, 100)
        dim = np.random.randint(1, 100)
        # some norm
        norm = "root_of_avg_of_spacedist_squared"
        pred = np.random.random((length, dim))
        meas = np.random.random((length, dim))
        error = measures.error_over_time(pred, meas, distance_measure="L2", normalization=norm)
        error_manually = np.linalg.norm(pred - meas, axis=1) / np.sqrt(np.mean(np.linalg.norm(meas, axis=1) ** 2))
        np.testing.assert_almost_equal(error, error_manually, decimal=15)

    @pytest.mark.skip(reason='measures.valid_time_index not yet implemented')
    def test_valid_time_index(self):
        error_series = np.linspace(0, 10, 11)
        epsilon = 5
        desired_valid_time_index = 6
        measured_valid_time_index = measures.valid_time_index(error_series, epsilon)
        np.testing.assert_equal(desired_valid_time_index, measured_valid_time_index)

    @pytest.mark.skip(reason='measures.valid_time_index not yet implemented')
    def test_valid_times_zero_error(self):
        error_series = np.zeros(5)
        epsilon = 0
        desired_valid_time_index = 4
        measured_valid_time_index = measures.valid_time_index(error_series, epsilon)
        np.testing.assert_equal(desired_valid_time_index, measured_valid_time_index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
