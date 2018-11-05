#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Make a mock cluster on the sky """

import numpy as np
import astropy.coordinates as coord
import astropy.units as u

__all__ = ['Cluster']


class Cluster(object):
    """class representing a mock star cluster"""
    def __init__(self, v0, sigmav, omegas=None, ws=None, k=0, b0=None):
        """
        Make a cluster with mean velocity `v0` and dispersion `sigma`

        v0 : array-like, (3,)
            Mean velocity vector in km/s
        sigmav : float
            (Isotropic) velocity dispersion in km/s
        omegas : array-like, (3,), optional
            Angular frequencies of rotation around x, y, z axis
        ws : array-like, (5,), optional
            Non-isotropic dilation
        k : float, optional
            Isotropic expansion (or contraction)
        b0 : array-like, (3,), optional
            Reference position vector where velocity field v = v0.
            Only matters when at least one of `omegas`, `ws` and `k` is non-zero.
        """
        # TODO option to accept astropy Quantities for parameters
        v0 = np.atleast_1d(v0)
        assert v0.shape == (3,), "v0 has a wrong shape"
        if b0 is not None:
            b0 = np.atleast_1d(b0)
            assert b0.shape == (3,), "b0 has a wrong shape"
        self.b0 = b0
        self.v0 = v0
        self.sigmav = sigmav
        if omegas:
            omegas = np.atleast_1d(omegas)
            assert omegas.shape == (3,), "omegas has a wrong shape"
        else:
            omegas = np.zeros(3)
        self.omegas = omegas
        if ws:
            ws = np.atleast_1d(ws)
            assert ws.shape == (5,), "ws has a wrong shape"
        else:
            ws = np.zeros(5)
        self.ws = ws
        self.k = k

        # construct tensor T from omegas, ws and k
        # velocity field at b is v(b) = v0 + T(b-b0)
        T = np.array([[ws[3], ws[2]-omegas[2], ws[1]+omegas[1]],
                      [ws[2]+omegas[2], ws[4], ws[0]-omegas[0]],
                      [ws[1]-omegas[1], ws[0]+omegas[0], 3*k-ws[3]-ws[4]]])
        self.T = T

    def __repr__(self):
        return "Cluster(b0={b0}, v0={v0}, sigmav={sigmav}".format(
            v0=list(self.v0), sigmav=self.sigmav
        )

    @classmethod
    def from_coord(cls, cluster_coord, sigmav, omegas=None, ws=None):
        """
        Make a cluster from astropy coordinates `cluster_coord`

        cluster_coord : astropy.coordinates.BaseCoordinateFrame instance
            coordinates containing cluster position and velocity

        The rest of the arguments are the same as `Cluster`.
        """
        if not isinstance(cluster_coord, coord.BaseCoordinateFrame):
            raise ValueError("`coord` must be an astropy coordinate instance.")
        if cluster_coord.shape:
            raise ValueError("`coord` must be a single cluster coordinate.")
        cluster_coord = cluster_coord.transform_to(coord.ICRS)

        b0 = cluster_coord.cartesian.xyz.to('pc').value
        v0 = cluster_coord.cartesian.differentials['s'].d_xyz.to(u.km/u.s).value
        return cls(b0, v0, sigmav, omegas=omegas, ws=ws)

    def sample(self, N=1):
        # position vector to each member (N, 3)
        bi = np.random.normal(loc=self.b0, scale=5, size=[N, 3])
        ui = self.v0 + np.zeros([N,3])   # simplification for now
        vi = np.random.normal(loc=ui, scale=self.sigmav)
        return bi, vi

    def sample_at(self, positions):
        """
        Sample cluster members at given `positions`

        positions : astropy.coordinates.BaseCoordinateFrame instance
            positions of member stars

        Returns ICRS coordinates with velocities populated
        """
        if not isinstance(positions, coord.BaseCoordinateFrame):
            raise ValueError("`positions` should be astropy coordinates")
        N = len(positions)
        icrs = positions.transform_to(coord.ICRS)
        bi = icrs.cartesian.xyz.to(u.pc).value
        assert bi.shape == (3, N), "WTF"
        if self.b0 is not None:
            bi -= self.b0[:, None]
        ui = self.v0 + np.random.normal([0, 0, 0], scale=self.sigmav, size=(N, 3))
        ui += np.einsum('ij,jN->Ni', self.T, bi)
        return coord.ICRS(*icrs.cartesian.xyz,
                          v_x=ui.T[0]*u.km/u.s, v_y=ui.T[1]*u.km/u.s, v_z=ui.T[2]*u.km/u.s,
                          representation_type='cartesian',
                          differential_type='cartesian')
