""" Make a mock cluster on the sky """
import numpy as np

from astropy import coordinates as coord, units as u

class Cluster(object):
    """class representing a mock star cluster"""
    def __init__(self, b0, v0, sigmav, omegas=None, ws=None, k=0):
        """

        b0 : array-like, (3,)
            A (random) reference position vector
        v0 : array-like, (3,)
            Mean velocity vector
        sigmav : float
            (Isotropic) velocity dispersion
        omegas :

        ws :

        """
        b0, v0 = np.atleast_1d(b0), np.atleast_1d(v0)
        assert b0.shape == (3,), "b0 has a wrong shape"
        assert v0.shape == (3,), "v0 has a wrong shape"
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
        T = np.array([[ws[3], ws[2]-omegas[2], ws[1]+omegas[1]],
                      [ws[2]+omegas[2], ws[4], ws[0]-omegas[0]],
                      [ws[1]-omegas[1], ws[0]+omegas[0], 3*k-ws[3]-ws[4]]])
        self.T = T

    def __repr__(self):
        return "Cluster(b0={b0}, v0={v0}, sigmav={sigmav}".format(
            b0=list(self.b0), v0=list(self.v0), sigmav=self.sigmav
        )

    @classmethod
    def from_coord(cls, cluster_coord, sigmav, omegas=None, ws=None):
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
