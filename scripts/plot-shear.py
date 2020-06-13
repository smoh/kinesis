"""
Script to illustrate the equivalence of shear and expansion.
"""
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import kinesis as kn
import gapipes as gp
import astropy.coordinates as coord
import astropy.units as u

kn.set_mpl_style()

T = np.random.uniform(size=(3, 3))
T = (T.T + T) / 2.0

b0 = np.array([10, 10, 10])
cl = kn.Cluster([0, 0, 0], 0.0, T=T, b0=b0)
# sample at grids
x, y, z = np.meshgrid(*[np.linspace(-5, 5, 5)] * 3)
x, y, z = x.ravel(), y.ravel(), z.ravel()
x, y, z = x + b0[0], y + b0[1], z + b0[2]
pos = coord.ICRS(x * u.pc, y * u.pc, z * u.pc, representation_type="cartesian")
cl.sample_at(pos)

g = cl.members.truth.g
v = g.icrs.velocity.d_xyz.value

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"})
ax1.quiver(x, y, z, v[0], v[1], v[2])

fig.savefig("../report/rotating-shear.png")
