#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import astropy.coordinates as coord
import astropy.units as u

from kinesis import mock
from kinesis import pipes

np.random.seed(73929)
#
# def test_sample_at():
#     cl = mock.Cluster([12.3, 23.1, -3.5], 0.5)
#     pos = coord.ICRS([21.3]*u.deg, [54]*u.deg, [95.2]*u.pc)
#     cl.sample_at(pos)

class TestCluster(object):
    def test_basic_cluster(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 2.5, 1000
        cl = mock.Cluster(v0, sigv).sample_sphere(N=N, Rmax=1)
        df = pipes.add_xv(cl.members.df, coord.ICRS)
        assert cl.N == N
        assert np.allclose(df[['vx', 'vy', 'vz']].std().values, sigv, .1)
        assert np.allclose(df[['vx', 'vy', 'vz']].mean().values, v0, .1)

        cl = mock.Cluster(v0, sigv).sample_sphere(N=N, Rmax=100)
        df = pipes.add_xv(cl.members.df, coord.ICRS)
        assert np.allclose(df[['vx', 'vy', 'vz']].std().values, sigv, .1)
        assert np.allclose(df[['vx', 'vy', 'vz']].mean().values, v0, .1)

    def test_sample_at(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 2.5, 1000
        pos = coord.ICRS(
            np.random.uniform(size=N)*np.pi*2*u.rad,
            (np.arccos(2*np.random.uniform(size=N)-1) - np.pi*0.5)*u.rad,
            distance=np.random.uniform(1, 100)*u.pc)
        cl = mock.Cluster(v0, sigv).sample_at(pos)
        df = pipes.add_xv(cl.members.df, coord.ICRS)
        assert np.allclose(df[['vx', 'vy', 'vz']].std().values, sigv, .1)
        assert np.allclose(df[['vx', 'vy', 'vz']].mean().values, v0, .1)

    def test_members_observe(self):
        v0, sigv, N = [-6.3, 45.2, 5.3], 2.5, 1000
        cl = mock.Cluster(v0, sigv)\
            .sample_sphere(N=N, Rmax=1)\
            .observe(cov=np.eye(3)*4)
        print(cl.members.data.columns.values)
        assert set(cl.members.data.columns) == \
            set(['ra', 'dec', 'parallax', 'pmra', 'pmdec',
                 'parallax_error', 'pmra_error', 'pmdec_error',
                 'parallax_pmra_corr', 'parallax_pmdec_corr',
                 'pmra_pmdec_corr'])
        assert (cl.members.data.pmra_error == 2).all()
        assert (cl.members.data.pmdec_error == 2).all()
        assert (cl.members.data.parallax_error == 2).all()
        assert (cl.members.data.parallax_pmra_corr == 0).all()
        assert (cl.members.data.parallax_pmdec_corr == 0).all()
        assert (cl.members.data.pmra_pmdec_corr == 0).all()

        import pandas as pd
        d = pd.read_csv("/Users/semyeong/projects/spelunky/oh17-dr2/dr2_vL_clusters_full.csv")\
            .groupby('cluster').get_group('Hyades')
        cl = mock.Cluster(v0, sigv)\
            .sample_sphere(N=N, Rmax=1)\
            .observe(error_from=d)
