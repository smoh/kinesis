"""
Script to make a plot demonstrating perspective effects.
"""
# This import registers the 3D projection, but is otherwise unused.
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


def add_sphere(ax=None):
    if ax is None:
        ax = plt.gca()
    # Make data
    u = np.linspace(0, np.deg2rad(360.0), 50)
    v = np.linspace(0, np.deg2rad(180.0), 50)
    x = 100 * np.outer(np.cos(u), np.sin(v))
    y = 100 * np.outer(np.sin(u), np.sin(v))
    z = 100 * np.outer(np.ones(np.size(u)), np.cos(v))
    # Plot the surface
    ax.plot_surface(x, y, z, cmap=cm.YlGnBu_r, antialiased=True, alpha=0.2)
    # ax.plot_wireframe(x, y, z, cmap=cm.coolwarm, antialiased=True)


def add_pqr(ra, dec, color, text, dxyz=np.zeros(3), ax=None):

    if ax is None:
        ax = plt.gca()
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    r = 100.0
    xyz_s = np.r_[
        [r * np.cos(dec) * np.cos(ra), r * np.cos(dec) * np.sin(ra), r * np.sin(dec)]
    ]

    phat = np.r_[[-np.sin(ra), np.cos(ra), 0.0]] * 12
    qhat = (
        np.r_[[-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]] * 12
    )
    rhat = np.r_[[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]] * 12

    ra_axis_line = np.vstack([xyz_s, xyz_s + phat * 2]).T
    dec_axis_line = np.vstack([xyz_s, xyz_s + qhat * 2]).T
    r_axis_line = np.vstack([xyz_s, xyz_s + rhat * 2]).T
    ax.plot(*ra_axis_line, color=color, lw=2)
    ax.plot(*dec_axis_line, color=color, lw=2)
    ax.plot(*r_axis_line, color=color, lw=2)

    ax.text(
        *(xyz_s + dxyz), text, color=color, size=20,
    )


def make_grid_cluster(ra, dec, distance, v0, degsize=5.0):
    b0 = coord.ICRS(ra * u.deg, dec * u.deg, d * u.pc).cartesian.xyz.value

    ra_bins = np.linspace(ra - degsize, ra + degsize, 11)
    dec_bins = np.linspace(dec - degsize, dec + degsize, 11)
    ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)
    ra_grid = ra_grid.ravel()
    dec_grid = dec_grid.ravel()
    memicrs = coord.ICRS(ra_grid * u.deg, dec_grid * u.deg, [d] * ra_grid.size * u.pc)

    cl = kn.Cluster(v0, 0.0, b0=b0).sample_at(memicrs)
    return cl


fig = plt.figure(figsize=(8, 8))
grid = fig.add_gridspec(7, 6)
ax = fig.add_subplot(grid[:4, 1:5], projection="3d")
ax.view_init(elev=30.0, azim=-52.0)
ax.set_xlim((-94, 94))
ax.set_ylim((-94, 94))
ax.set_zlim((-94, 94))
ax.set_xticks(np.linspace(-80, 80, 5))
ax.set_yticks(np.linspace(-80, 80, 5))
ax.set_zticks(np.linspace(-80, 80, 5))
ax.set_xlabel("$x$",)
ax.set_ylabel("$y$",)
ax.set_zlabel("$z$",)

axes = [
    fig.add_subplot(grid[4:6, :2]),
    fig.add_subplot(grid[4:6, 2:4]),
    fig.add_subplot(grid[4:6, 4:]),
]

for cax in axes:
    cax.set(xlabel="R.A. [deg]")
    cax.tick_params(direction="in", labelsize=11)
axes[0].set_ylabel("Decl. [deg]")
ax.grid(False)


ax2 = fig.add_subplot(grid[6, :2],)
ax2.set(ylabel=r"$v_\delta\,[\mathrm{km}\,\mathrm{s}^{-1}]$")

# add box indicating one Decl row and arrow that connect to new axes.
axes[0].plot([40, 50], [40, 40], lw=4, alpha=0.5, zorder=-5, color="tab:blue")
patch = ConnectionPatch(
    xyA=(39.8, 40),
    xyB=(0, 1),
    coordsA="data",
    coordsB="axes fraction",
    axesA=axes[0],
    axesB=ax2,
    arrowstyle="->",
    connectionstyle="arc3,rad=.5",
    clip_on=False,
)
axes[0].add_artist(patch)

# three positions:
ra = np.array([45, 300, 340.0])
dec = np.array([45, 45.0, -65.0])
d = 100  # pc

# make velocity vector radial at position 1
b0tmp = coord.ICRS(
    ra[0] * u.deg,
    dec[0] * u.deg,
    d * u.pc,
    0.0 * u.mas / u.yr,
    0.0 * u.mas / u.yr,
    10 * u.km / u.s,
)
v0 = b0tmp.velocity.d_xyz.value
print(v0)

params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}
plt.rcParams.update(params)
fig.suptitle(
    r"$d = 100~\mathrm{pc}\,\,|\boldsymbol{v}_0| = 10~\mathrm{km}\,\mathrm{s}^{-1}$"
)

add_sphere(ax)

colors = ["#2269c4", "#c4451c", "#439064"]
poslabels = ["1", "2", "3"]
# offset to add to text labeling each position
dxyzs = np.array([[1.0, 2.0, -1.0], [-1, -0.5, -1], [-1, -1, 0]]) * 10

for ind, (cra, cdec, color, cax, l, dxyz) in enumerate(
    zip(ra, dec, colors, axes, poslabels, dxyzs)
):

    cax.set_title(
        r"{}: $(\alpha,\,\delta)=({:.0f},\,{:.0f})$".format(l, cra, cdec),
        fontdict=dict(fontsize=13, color=color),
    )

    add_pqr(cra, cdec, color, l, ax=ax, dxyz=dxyz)

    xyz = (
        np.array(
            [
                np.cos(np.deg2rad(cdec)) * np.cos(np.deg2rad(cra)),
                np.cos(np.deg2rad(cdec)) * np.sin(np.deg2rad(cra)),
                np.sin(np.deg2rad(cdec)),
            ]
        )
        * 100.0
    )
    ax.quiver(*xyz, *v0, length=5, color="black")

    cl = make_grid_cluster(cra, cdec, d, v0)
    c = cl.members.truth.g
    b0 = coord.ICRS(cra * u.deg, cdec * u.deg, d * u.pc).cartesian.xyz.value
    cc = coord.ICRS(
        *(b0 * u.pc),
        *(v0 * u.km / u.s),
        representation_type=coord.CartesianRepresentation,
        differential_type=coord.CartesianDifferential,
    )
    # NOTE:cos(dec) factor is not applied when differential is accessed this way.
    vra0 = (
        cc.spherical.differentials["s"].d_lon.value
        * d
        / 1e3
        * gp.accessors._tokms
        * np.cos(np.deg2rad(cdec))
    )
    vdec0 = cc.spherical.differentials["s"].d_lat.value * d / 1e3 * gp.accessors._tokms
    cax.quiver(
        c.icrs.ra.value, c.icrs.dec.value, c.vra.values - vra0, c.vdec.values - vdec0,
    )

    # highlight a slice at constant decl for second order effect
    if ind == 0:
        sel = slice(11)
        ax2.scatter(c.icrs.ra.value[sel], (c.vdec.values - vdec0)[sel])

for cax in axes:
    xlim, ylim = cax.get_xlim(), cax.get_ylim()
    cax.set_xlim(np.array(xlim) + np.r_[[-1, 1]])
    cax.set_ylim(np.array(ylim) + np.r_[[-1, 1]])

fig.tight_layout(rect=[None, None, None, 1], h_pad=1, w_pad=0.2)
fig.savefig("fig1.pdf", dpi=120)
