# -*- coding: utf-8 -*-
""" Tests if the rescomp.simulations module works as it should """

import unittest
import pytest
import numpy as np
import rescomp


@pytest.mark.xfail(reason='the KS simulation sometimes diverges using the numpy FFT')
class testSimulations(unittest.TestCase):
    def test_kuramoto_sivashinski_40d_22l_05t_divergence(self):
        ks_sys_flag = 'kuramoto_sivashinsky'
        dimensions = 40
        system_size = 22
        dt = 0.5
        time_steps = 1000
        sim_data = rescomp.simulate_trajectory(
            sys_flag=ks_sys_flag, dimensions=dimensions, system_size=system_size, dt=dt,
            time_steps=time_steps)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100


class testKuramotSivashinskiVariants40d22l05t(unittest.TestCase):
    def setUp(self):
        self.ks_sys_flag = 'kuramoto_sivashinsky_custom'
        self.dimensions = 40
        self.system_size = 22
        self.dt = 0.5
        self.time_steps = 1000

    @pytest.mark.xfail(reason='the KS simulation sometimes diverges using the numpy FFT')
    def test_kuramoto_sivashinski_40d_22l_05t_npfft_64bit_divergence(self):
        precision = 64
        fft_type = 'numpy'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    @pytest.mark.xfail(reason='the KS simulation sometimes diverges using the numpy FFT')
    def test_kuramoto_sivashinski_40d_22l_05t_npfft_32bit_divergence(self):
        precision = 32
        fft_type = 'numpy'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_40d_22l_05t_scfft_64bit_divergence(self):
        pytest.importorskip("scipy")
        precision = 64
        fft_type = 'scipy'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_40d_22l_05t_scfft_32bit_divergence(self):
        pytest.importorskip("scipy")
        precision = 32
        fft_type = 'scipy'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_40d_22l_05t_pyfftw_np_64bit_divergence(self):
        pytest.importorskip("pyfftw")
        precision = 64
        fft_type = 'pyfftw_np'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_40d_22l_05t_pyfftw_np_32bit_divergence(self):
        pytest.importorskip("pyfftw")
        precision = 32
        fft_type = 'pyfftw_np'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_40d_22l_05t_pyfftw_sc_64bit_divergence(self):
        pytest.importorskip("pyfftw")
        precision = 64
        fft_type = 'pyfftw_sc'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_40d_22l_05t_pyfftw_sc_32bit_divergence(self):
        pytest.importorskip("pyfftw")
        precision = 32
        fft_type = 'pyfftw_sc'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_40d_22l_05t_pyfftw_fftw_64bit_divergence(self):
        pytest.importorskip("pyfftw")
        precision = 64
        fft_type = 'pyfftw_fftw'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100

    def test_kuramoto_sivashinski_40d_22l_05t_pyfftw_fftw_32bit_divergence(self):
        pytest.importorskip("pyfftw")
        precision = 32
        fft_type = 'pyfftw_fftw'
        sim_data = rescomp.simulate_trajectory(
            sys_flag=self.ks_sys_flag, dimensions=self.dimensions, system_size=self.system_size,
            dt=self.dt, time_steps=self.time_steps, precision=precision, fft_type=fft_type)
        assert not np.isnan(sim_data).any()
        assert np.amax(np.abs(sim_data)) < 100


if __name__ == "__main__":
    unittest.main(verbosity=2)
