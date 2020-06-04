"""Demonstrate perspective rotation/shear/expansion."""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
import kinesis as kn
import gapipes as gp

v0 = [-6.3, 45.2, 5.3]
v0 = [-6.3, 5.2, -45.2]
sigmav = 0.0
N = 100
d = 100
b0 = coord.ICRS(120 * u.deg, 25 * u.deg, d * u.pc).cartesian.xyz.value

# # get cartesian v0 with zero proper motions and only radial velocity
# b0tmp = coord.ICRS(
#     120 * u.deg,
#     25 * u.deg,
#     d * u.pc,
#     0.0 * u.mas / u.yr,
#     0.0 * u.mas / u.yr,
#     10 * u.km / u.s,
# )
# v0 = b0tmp.velocity.d_xyz.value
# print(v0)


Rmax = 10  # pc
print(b0)
# coordinate object for the cluster center with velocities
# NOTE: astropy coordinates do not allow to mix spherical X with cartesian V!
cc = coord.ICRS(
    *(b0 * u.pc),
    *(v0 * u.km / u.s),
    representation_type=coord.CartesianRepresentation,
    differential_type=coord.CartesianDifferential
)
vra0 = cc.spherical.differentials["s"].d_lon.value * d / 1e3 * 4.74
vdec0 = cc.spherical.differentials["s"].d_lat.value * d / 1e3 * 4.74

# TODO: sample plane normal to b0 -- this will make it very clear I think...
cl = kn.Cluster(v0, sigmav, b0=b0).sample_sphere(N=N, Rmax=Rmax)

# hy = pd.read_csv("../data/hyades_full.csv").groupby('in_dr2').get_group(True)
# v0 = [-6.3, 45.2, 5.3]
# b0 = np.array([17.15474298, 41.28962638, 13.69105771])
# cl = kn.Cluster(v0, sigmav, b0=b0).sample_at(hy.g.icrs)
# vra0 = cl.icrs.pm_ra_cosdec.value * cl.icrs.distance.value /1e3*4.74
# vdec0 = cl.icrs.pm_dec.value * cl.icrs.distance.value /1e3*4.74

c = cl.members.truth.g


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
q1 = ax1.quiver(c.icrs.cartesian.x, c.icrs.cartesian.y, c.vra, c.vdec)
ax1.quiverkey(q1, 0.9, 0.9, 50, "50", coordinates="axes")
q2 = ax2.quiver(c.icrs.cartesian.x, c.icrs.cartesian.y, c.vra - vra0, c.vdec - vdec0)
ax2.quiverkey(q2, 0.9, 0.9, 10, "10", coordinates="axes")

# radius_deg = 5
# ra_grid = np.linspace()


#%%
from scipy.spatial.transform import Rotation as R


def sample_circle(N=1, R=1):
    theta = np.random.uniform(size=N) * 2 * np.pi
    r = R * np.sqrt(np.random.uniform(size=N))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack([x, y]).T


def sample_surface(x0=None, Rdeg=1, N=1):
    """
    Rdegree (float): radius of patch in degrees
    """
    Rrad = np.deg2rad(Rdeg)
    assert np.shape(x0) == (3,), "`x0` has a wrong shape."

    r, phi_prime, theta = coord.cartesian_to_spherical(
        *x0
    )  # Returned angles are in radians.
    r = r.value
    phi = np.pi / 2.0 - phi_prime.value  # to usual polar angle
    theta = theta.value

    thetas = np.random.uniform(low=theta - Rrad, high=theta + Rrad, size=N)

    # phi = arccos (nu); nu is uniform
    nu1, nu2 = np.cos(phi - Rrad), np.cos(phi + Rrad)
    nus = np.random.uniform(low=nu1, high=nu2, size=N)
    phis = np.cos(nus)
    xyz = coord.spherical_to_cartesian(r, np.pi / 2.0 - phis, thetas)
    return np.vstack(list(map(lambda x: x.value, xyz))).T


from mpl_toolkits.mplot3d import axes3d

xi, yi, zi = sample_surface([10, -10, 10], 5, N=100).T
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})  # , 'aspect':'equal'})
phi = np.linspace(0, np.pi, 10)
theta = np.linspace(0, 2 * np.pi, 20)
x = np.outer(np.sin(theta), np.cos(phi)) * 10 * np.sqrt(3)
y = np.outer(np.sin(theta), np.sin(phi)) * 10 * np.sqrt(3)
z = np.outer(np.cos(theta), np.ones_like(phi)) * 10 * np.sqrt(3)
ax.plot_wireframe(x, y, z, color="k", rstride=1, cstride=1)
ax.scatter(xi, yi, zi, s=100, c="r", zorder=10)


# xy = sample_circle(N=1000).T
# fig, ax = plt.subplots()
# ax.scatter(*xy)
