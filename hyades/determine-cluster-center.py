"""
Determine cluster center with a radius cut as the position
where the postional membership within the radius cut does not change.

- The radius cut should be large enough to contain substantial number of stars
otherwise it will just depend on the statistical fluctuation of the mean of
small number of stars.
- The radius cut should be not too large as then, it
will be subject to larger contamination rate far from the cluster.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.coordinates as coord

import kinesis as kn

kn.set_mpl_style()


def xyz_icrs_to_galactic(xyz):
    c = coord.ICRS(*xyz, representation_type="cartesian")
    return c.transform_to(coord.Galactic).cartesian.xyz.value


df = kn.data.load_hyades_dataset()
df = df.loc[df['Member_r19']!='other'].reset_index(drop=True)
print(len(df), 'rows')
cl_gaia = df.loc[df["in_dr2"] == True].copy()

b_c_icrs_cl_gaia_mean = cl_gaia.g.icrs.cartesian.xyz.mean(axis=1).value
b_c_galactic_cl_gaia_mean = xyz_icrs_to_galactic(b_c_icrs_cl_gaia_mean)
xyz = df.g.icrs.cartesian.xyz.value

# list storage for boolean flag array; initially consider all stars
in_rcut = [np.ones(xyz.shape[1]).astype(bool)]
# list storage for boolean flag array; initially set with Gaia Collab DR2 sample
r_cut = 10  # pc, radius cut
niter = 20  # maximum number of iteration
print("N={:d} r_cut={:.2f} max iter={:d}".format(len(df), r_cut, niter))
for i in range(niter):
    prev = in_rcut[-1]
    b_c = xyz[:, prev].mean(axis=1)
    r_c = np.linalg.norm(xyz - b_c[:, None], axis=0)
    current = r_c < r_cut
    bool_remove = (~current) & (prev)
    bool_include = (current) & (~prev)
    if (current == prev).all():
        print("iter {:2d} b_c={} membership converged".format(i, b_c))
        break
    else:
        print(
            "iter {:2d} b_c={} removing {:d} including {:d}".format(
                i, b_c, bool_remove.sum(), bool_include.sum()
            )
        )
        in_rcut.append(current)

# report final values
b_c_icrs_iter_mean = b_c
b_c_galactic_iter_mean = xyz_icrs_to_galactic(b_c)
r_c = np.linalg.norm(xyz - b_c_icrs_iter_mean[:, None], axis=0)
n_r_cut = (r_c < r_cut).sum()
n_r_cut_p05 = (r_c < r_cut + 0.5).sum()
n_r_cut_m05 = (r_c < r_cut - 0.5).sum()
print("final b_c icrs     =", b_c_icrs_iter_mean)
print("final b_c galactic =", b_c_galactic_iter_mean)
print(n_r_cut, n_r_cut_p05, n_r_cut_m05)

reino2018 = dict(
    b_c_galactic=np.array([-44.16, 0.66, -17.76]),  # pc
    b_c_galactic_err=np.array([0.74, 0.39, 0.41]),  # pc
)

report_df = pd.DataFrame.from_dict(
    {
        "final b_c icrs": b_c_icrs_iter_mean,
        "final b_c galactic": b_c_galactic_iter_mean,
        "mean of cl gaia icrs": b_c_icrs_cl_gaia_mean,
        "mean of cl gaia galactic": b_c_galactic_cl_gaia_mean,
        "Reino 2018 galactic": reino2018["b_c_galactic"],
    },
    orient="index",
    columns=("x", "y", "z"),
)
print(report_df)


#%% summary plot: distribution of stars from the center
fig, (axhist, axdens) = plt.subplots(1, 2, figsize=(7, 3), sharex=True)
bins = np.logspace(-1, 2.5, 32)
axhist.hist(r_c, bins)
axhist.axvline(r_cut, c="k")
axhist.set_xlabel("$r_c$ [pc]")
axhist.set_ylabel("count [pc]")
axhist.set_xscale("log")

s, be = np.histogram(r_c, bins)
bc = (be[1:] + be[:-1]) * 0.5
numdens = s / (np.pi * 4 * bc ** 2) / (be[1] - be[0])
axdens.plot(bc, numdens, "o-")
axdens.axvline(r_cut, c="k")
axdens.set_xscale("log")
axdens.set_yscale("log")
axdens.set_ylim(1e-3, 200)
axdens.set_xlabel("$r_c$ [pc]")
axdens.set_ylabel("number density [pc$^{-3}$]")
fig.tight_layout()
fig.savefig("../report/r_c_dist.pdf")

#%% summary plot: distribution of stars in Galactic coordinates
fig, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(3, 6), sharex=True,)
ax_xy.set_aspect("equal")
ax_xz.set_aspect("equal")
ax_xy.scatter(df["gx"], df["gy"], s=1, c="tab:gray")
ax_xz.scatter(df["gx"], df["gz"], s=1, c="tab:gray")


def add_circle(center, radius, ax=None):
    from matplotlib.patches import Circle

    if ax is None:
        ax = plt.gca()
    circle = Circle(center, radius, facecolor="None", edgecolor="k")
    ax.add_patch(circle)


xy_cen = [b_c_galactic_iter_mean[0], b_c_galactic_iter_mean[1]]
xz_cen = [b_c_galactic_iter_mean[0], b_c_galactic_iter_mean[2]]
add_circle(xy_cen, r_cut, ax=ax_xy)
add_circle(xz_cen, r_cut, ax=ax_xz)

ax_xy.set_xlabel("$x$ [pc]")
ax_xy.set_ylabel("$y$ [pc]")

ax_xz.set_xlabel("$x$ [pc]")
ax_xz.set_ylabel("$z$ [pc]")
fig.suptitle("Galactic")
fig.tight_layout()
fig.savefig("../report/galactic_xyz.pdf")

N_rv = (df['radial_velocity'].notna()).sum()
N_10 = (r_c<10).sum()
N_10_rv = ((r_c<10) & (df['radial_velocity'].notna())).sum()
print('N rv =', N_rv)
print('N(r_c<10) = {} N(r_c<10 and has rv) = {}'.format(N_10, N_10_rv))
print('N(r_c>10) = {} N(r_c<10 and has rv) = {}'.format(len(df)-N_10, N_rv-N_10_rv))