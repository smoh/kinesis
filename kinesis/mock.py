#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Make a mock cluster on the sky """

import numpy as np
import pandas as pd

import astropy.coordinates as coord
import astropy.units as u

__all__ = ['sample_uniform_sphere', 'Cluster']


def sample_uniform_sphere(x0=None, Rmax=1, N=1, return_icrs=False):
    """ Sample points within a uniform density sphere

    x0 : array-like, (3,)
        mean position of the sphere
    Rmax : float, optional
        maximum radius of the sphere
    N : int, optional
        number of points to sample

    Returns (3, N) array of xyz positions
    """
    rhat = np.random.normal(size=(3, N))
    rhat /= np.linalg.norm(rhat, axis=0)
    xyz = np.random.uniform(size=(1, N))**(1./3.) * Rmax * rhat
    if x0 is not None:
        x0 = np.atleast_1d(x0)
        assert x0.shape == (3,), "`x0` has a wrong shape"
        xyz = xyz + x0[:, None]
    if return_icrs:
        return coord.ICRS(*xyz*u.pc, representation_type='cartesian')
    return xyz


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

        Attributes
        ----------
        members : None or `ClusterMembers` instance
            Use sample_
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
        self.members = None

    def __repr__(self):
        return "Cluster(b0={b0}, v0={v0}, sigmav={sigmav})".format(
            b0=list(self.b0), v0=list(self.v0), sigmav=self.sigmav
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

    def sample_sphere(self, N=1, Rmax=10):
        if self.members:
            raise AttributeError('The cluster already have members.')
        xyz_coords = sample_uniform_sphere(x0=self.b0, Rmax=Rmax, N=N, return_icrs=True)
        return self.sample_at(xyz_coords)

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
        coordinates = coord.ICRS(
            *icrs.cartesian.xyz,
            v_x=ui.T[0]*u.km/u.s, v_y=ui.T[1]*u.km/u.s, v_z=ui.T[2]*u.km/u.s,
            representation_type='cartesian',
            differential_type='cartesian')
        self.members = ClusterMembers(coordinates)
        return self

    @property
    def N(self):
        """ Size of the cluster """
        if self.members is None:
            return 0
        return self.members.N


class ClusterMembers(object):
    """ class representing members of a Mock cluster """

    def __init__(self, coordinates):
        """
        Initialize cluster members with coordinates

        coordinates : astropy.coordinates.BaseCoordinateFrame instance
            must contain positions and velocities

        Attributes
        ----------
        N : int
            number of members
        icrs : astropy.coordinates.ICRS instance
            ICRS coordinates with spherical representation

        """
        if hasattr(coordinates, 'differentials'):
            raise ValueError("`coordinates` should have differentials")
        self.N = len(coordinates)

        assert coordinates.representation_type == coord.CartesianRepresentation
        assert coordinates.differential_type == coord.CartesianDifferential
        sph = coordinates.spherical
        # make a Gaia DataFrame
        df = pd.DataFrame({
            'ra': sph.lon.deg,
            'dec': sph.lat.deg,
            'parallax': 1/sph.distance.to(u.kpc).value,
            # NOTE: `d_lon` should be multiplied by cos(`lat`)!
            'pmra': sph.differentials['s'].d_lon.to(u.mas/u.yr).value * np.cos(sph.lat.rad),
            'pmdec': sph.differentials['s'].d_lat.to(u.mas/u.yr).value,
            'radial_velocity': sph.differentials['s'].d_distance.to(u.km/u.s).value
        })
        self.df = df

        # To have natural attributes of ICRS, make it again with spherical
        # representation
        self.icrs = coord.ICRS(
            df.ra.values*u.deg, df.dec.values*u.deg,
            distance=1e3*u.pc/df.parallax.values,
            pm_ra_cosdec=df.pmra.values*u.mas/u.yr,
            pm_dec=df.pmdec.values*u.mas/u.yr,
            radial_velocity=df.radial_velocity.values*u.km/u.s
        )

        # def add_errors(self)
