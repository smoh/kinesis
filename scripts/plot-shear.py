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

# # generate a random symmetrix matrix
# T = np.random.uniform(size=(2, 2))
# T = (T.T + T) / 2.0

T = np.array([[0.65896236, 0.78559313], [0.78559313, 0.33118707]])

# sample at grids
x, y = np.meshgrid(*[np.linspace(-5, 5, 11)] * 2)
x, y = x.ravel(), y.ravel()
X = np.vstack([x, y])
V = np.einsum("ji,in->jn", T, X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

ax1.quiver(*X, *V)

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

first_axis_color = "tab:red"
second_axis_color = "tab:blue"

ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set(linewidth=2, color=first_axis_color)
ax2.spines["left"].set(linewidth=2, color=second_axis_color)
ax2.set_xticks([])
ax2.set_yticks([])

# rotate to frame defined by principal axes of shear
W, R = np.linalg.eig(T)
invR = R.T

# indicate principal axes
length = 2
ax1.plot([0, R[0, 0] * length], [0, R[1, 0] * length], color=first_axis_color, lw=2)
ax1.plot([0, R[0, 1] * length], [0, R[1, 1] * length], color=second_axis_color, lw=2)

# Xnew = np.einsum("ij,jn->in", invR, X)
# Vnew = np.einsum("ij,jn->in", invR, V)
# ax2.quiver(*Xnew, *Vnew)


x, y = np.meshgrid(*[np.linspace(-5, 5, 11)] * 2)
x, y = x.ravel(), y.ravel()
Xnew = np.vstack([x, y])
Vnew = np.einsum("ji,in->jn", W * np.eye(2), Xnew)
Vnew = np.einsum("ji,in->jn", invR @ T @ R, Xnew)
ax2.quiver(*Xnew, *Vnew)

ax1.axis([-7, 7, -7, 7])
ax2.axis([-7, 7, -7, 7])

fig.tight_layout()
fig.savefig("../report/rotating-shear.pdf")
