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

        meas = np.random.random((length, dim))
        pred = np.random.random((length, dim))

        norm = pred.shape[0]
        rmse_desired = np.sqrt(((pred - meas) ** 2).sum() / norm)

        rmse = measures.rmse(meas, pred)

        # results not exactly equal due to numpy optimizations
        np.testing.assert_allclose(rmse, rmse_desired, rtol=1e-15)

    def test_nrmse(self):
        length = np.random.randint(10, 100)
        dim = np.random.randint(10, 100)

        meas = np.random.random((length, dim))
        pred = np.random.random((length, dim))

        norm = (meas ** 2).sum()
        nrmse_desired = np.sqrt(((pred - meas) ** 2).sum() / norm)

        nrmse = measures.nrmse(meas, pred)

        # results not exactly equal due to numpy optimizations
        np.testing.assert_allclose(nrmse, nrmse_desired, rtol=1e-15)

        # np.testing.assert_equal(np.sqrt((meas ** 2).sum()), np.linalg.norm(meas))

    def test_nrmse_over_time(self):
        length = 1
        dim = np.random.randint(10, 100)

        meas = np.random.random((length, dim))
        pred = np.random.random((length, dim))

        meh = meas[0]

        nrmse_desired = measures.nrmse(meas, pred)
        nrmse = measures.nrmse_over_time(meas, pred)

        np.testing.assert_equal(nrmse, nrmse_desired)



if __name__ == "__main__":
    unittest.main(verbosity=2)
