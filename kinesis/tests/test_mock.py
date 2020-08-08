#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import astropy.coordinates as coord
import astropy.units as u

from kinesis import mock
import gapipes as pipes

np.random.seed(73929)


def test_sample_uniform_sphere():
    assert mock.sample_uniform_sphere().shape == (3, 1)

    r = mock.sample_uniform_sphere(return_icrs=True)
    assert isinstance(r, coord.ICRS)

    r = mock.sample_uniform_sphere(x0=[1, 2, 3], Rmax=10.0, N=10)
    assert r.shape == (3, 10)

    with pytest.raises(AssertionError):
        mock.sample_uniform_sphere(x0=0)


class TestCluster(object):
    def test_basic_cluster(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 2.5, 1000
        cl = mock.Cluster(v0, sigv).sample_sphere(N=N, Rmax=1)
        df = pipes.add_xv(cl.members.truth, coord.ICRS)
        assert cl.N == N
        assert np.allclose(df[["vx", "vy", "vz"]].std().values, sigv, 0.1)
        assert np.allclose(df[["vx", "vy", "vz"]].mean().values, v0, 0.1)

        cl = mock.Cluster(v0, sigv).sample_sphere(N=N, Rmax=100)
        df = pipes.add_xv(cl.members.truth, coord.ICRS)
        assert np.allclose(df[["vx", "vy", "vz"]].std().values, sigv, 0.1)
        assert np.allclose(df[["vx", "vy", "vz"]].mean().values, v0, 0.1)
    
    def test_T_nonzero(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 0.0, 1000
        cl = mock.Cluster(v0, sigv, omegas=[0,0,50.0], b0=[30, 40, 50])
        assert np.array_equal(cl.T, np.array([[0, -50, 0], [50, 0, 0], [0, 0, 0]]))
    
    def test_T_nonzero_errors(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 2.5, 1000
        with pytest.raises(ValueError):
            mock.Cluster(v0, sigv, T=np.eye(3))
        
        with pytest.raises(ValueError):
            mock.Cluster(v0, sigv, omegas=[1, 0, 0])
        
        with pytest.raises(ValueError) as e:
            mock.Cluster(v0, sigv, b0=[0,0,0], omegas=[1,0,0], T=np.eye(3))
        assert e.value.args[0] == "Either set (omegas, ws, k) or matrix `T`; do not set both."

        # Give wrong-shape parameters.
        with pytest.raises(AssertionError):
            mock.Cluster(v0, sigv, b0=[0,0,0], T=[1,0,0,0])
        with pytest.raises(AssertionError):
            mock.Cluster(v0, sigv, b0=[0,0,0], omegas=[1,0,0,0])
        with pytest.raises(AssertionError):
            mock.Cluster(v0, sigv, b0=[0,0,0], ws=[1,0,0,0])

    def test_sample_at(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 2.5, 1000
        pos = coord.ICRS(
            np.random.uniform(size=N) * np.pi * 2 * u.rad,
            (np.arccos(2 * np.random.uniform(size=N) - 1) - np.pi * 0.5) * u.rad,
            distance=np.random.uniform(1, 100) * u.pc,
        )
        cl = mock.Cluster(v0, sigv).sample_at(pos)
        df = pipes.add_xv(cl.members.truth, coord.ICRS)
        assert np.allclose(df[["vx", "vy", "vz"]].std().values, sigv, 0.1)
        assert np.allclose(df[["vx", "vy", "vz"]].mean().values, v0, 0.1)

        # anisotropic case 1
        v0, sigv, N = [-6.3, 45.2, 5.3], [2.5, 5, 1], 1000
        pos = coord.ICRS(
            np.random.uniform(size=N) * np.pi * 2 * u.rad,
            (np.arccos(2 * np.random.uniform(size=N) - 1) - np.pi * 0.5) * u.rad,
            distance=np.random.uniform(1, 100) * u.pc,
        )
        cl = mock.Cluster(v0, sigv).sample_at(pos)
        df = pipes.add_xv(cl.members.truth, coord.ICRS)
        assert np.allclose(df[["vx", "vy", "vz"]].std().values, sigv, 0.1)
        assert np.allclose(df[["vx", "vy", "vz"]].mean().values, v0, 0.1)

        # anisotropic case 2; give covariance matrix
        v0, sigv, N = [-6.3, 45.2, 5.3], np.eye(3), 1000
        pos = coord.ICRS(
            np.random.uniform(size=N) * np.pi * 2 * u.rad,
            (np.arccos(2 * np.random.uniform(size=N) - 1) - np.pi * 0.5) * u.rad,
            distance=np.random.uniform(1, 100) * u.pc,
        )
        cl = mock.Cluster(v0, sigv).sample_at(pos)
        df = pipes.add_xv(cl.members.truth, coord.ICRS)
        assert np.allclose(df[["vx", "vy", "vz"]].cov().values, sigv, atol=0.1)
        assert np.allclose(df[["vx", "vy", "vz"]].mean().values, v0, atol=0.1)

    def test_members_observe(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 2.5, 1000
        cl = (
            mock.Cluster(v0, sigv).sample_sphere(N=N, Rmax=1).observe(cov=np.eye(3) * 4)
        )
        assert set(cl.members.observed.columns) == set(
            [
                "ra",
                "dec",
                "parallax",
                "pmra",
                "pmdec",
                "parallax_error",
                "pmra_error",
                "pmdec_error",
                "parallax_pmra_corr",
                "parallax_pmdec_corr",
                "pmra_pmdec_corr",
            ]
        )
        assert (cl.members.observed.pmra_error == 2).all()
        assert (cl.members.observed.pmdec_error == 2).all()
        assert (cl.members.observed.parallax_error == 2).all()
        assert (cl.members.observed.parallax_pmra_corr == 0).all()
        assert (cl.members.observed.parallax_pmdec_corr == 0).all()
        assert (cl.members.observed.pmra_pmdec_corr == 0).all()

        # Feed dataframe to test errors_from
        cl2 = (
            mock.Cluster(v0, sigv)
            .sample_sphere(N=N, Rmax=1)
            .observe(error_from=cl.members.observed)
        )
        assert (cl2.members.observed.pmra_error == 2).all()
        assert (cl2.members.observed.pmdec_error == 2).all()
        assert (cl2.members.observed.parallax_error == 2).all()
        assert (cl2.members.observed.parallax_pmra_corr == 0).all()
        assert (cl2.members.observed.parallax_pmdec_corr == 0).all()
        assert (cl2.members.observed.pmra_pmdec_corr == 0).all()

    def test_rv_error(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 2.5, 1000
        cl = (
            mock.Cluster(v0, sigv)
            .sample_sphere(N=N, Rmax=1)
            .observe(cov=np.eye(3) * 4, rv_error=np.array([1] * 999 + [np.nan]))
        )
        assert "radial_velocity" in cl.members.observed.columns
        assert "radial_velocity_error" in cl.members.observed.columns
        assert np.isnan(cl.members.observed["radial_velocity"].values[-1])
        assert np.isnan(cl.members.observed["radial_velocity_error"].values[-1])
