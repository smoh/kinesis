# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% {"tags": ["setup"]}
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import cartopy.crs as ccrs

import pandas as pd
import numpy as np
import scipy as sp
from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
import arviz as az
import seaborn as sns

import kinesis as kn
import gapipes as gp

# %% {"tags": ["setup"]}
plt.style.use("kinesis.mplstyle")

# %%
# %store -r out_full
df = out_full.loc[out_full['Member_r19']!='other']

# %%
print(f"{len(df)} rows, {len(df.columns)} columns")

# %%
# slices of data
gdr2 = df.groupby('in_dr2').get_group(True)

# %%
df[["in_dr2", "in_leao", "in_meingast", "in_roser"]].fillna(False).groupby(["in_dr2"]).sum()

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), subplot_kw=dict(projection=ccrs.Mollweide()))
ax.gridlines(
    crs=ccrs.Geodetic(),
    xlocs=[-180, -90, 0, 90, 180],
    ylocs=[0, 45, 90, -45, -90],
    linewidth=0.5,
    zorder=0,
)
ax.scatter(df["ra"], df["dec"], s=1, c='k', transform=ccrs.Geodetic())
ax.scatter(gdr2["ra"], gdr2["dec"], s=1, transform=ccrs.Geodetic())
ax.set_global()
ax.set_title("Sky distribution")
fig.tight_layout()
fig.savefig('../plots/hyades-sky.pdf')

# %%
fig, ax = plt.subplots(1, 1,  figsize=(4, 2.5),subplot_kw=dict(projection=ccrs.Mollweide(central_longitude=180)))
ax.gridlines(
    crs=ccrs.Geodetic(),
    xlocs=[-180, -90, 0, 90, 180],
    ylocs=[0, 45, 90, -45, -90],
    linewidth=0.5,
    zorder=0,
)
ax.scatter(df["l"], df["b"], s=1, c='k', transform=ccrs.Geodetic())
ax.scatter(gdr2["l"], gdr2["b"], s=1, transform=ccrs.Geodetic())
ax.set_global()
ax.set_title("Galactic (centered on $l=180$)")
fig.tight_layout()
fig.savefig('../plots/hyades-galactic-distribution.pdf')

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for cax in ax:
    cax.set_aspect("equal")
for dset, color in zip([df, gdr2], ["k", None]):
    cartx, cartv = dset.g.icrs.cartesian, dset.g.icrs.velocity
    ax[0].scatter(cartx.x, cartx.y, s=1, c=color)
    ax[1].scatter(cartx.x, cartx.z, s=1, c=color)
for cax in ax:
    cax.set_xlabel("$X_\mathrm{ICRS}$")
ax[0].set_ylabel("$Y_\mathrm{ICRS}$")
ax[1].set_ylabel("$Z_\mathrm{ICRS}$")
fig.tight_layout()
fig.savefig('../plots/hyades-xyz-icrs.pdf')

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for cax in ax: cax.set_aspect('equal');
for dset, color in zip([df, gdr2], ['k', None]):
    cartx, cartv = dset.g.galactic.cartesian, dset.g.galactic.velocity
    ax[0].scatter(cartx.x, cartx.y, s=1, c=color,);
    ax[1].scatter(cartx.x, cartx.z, s=1, c=color,);
for cax in ax: cax.set_xlabel('$X_\mathrm{Galactic}$')
ax[0].set_ylabel('$Y_\mathrm{Galactic}$')
ax[1].set_ylabel('$Z_\mathrm{Galactic}$');
fig.tight_layout()
fig.savefig('../plots/hyades-xyz-galactic.pdf')

# %%
df[["radial_velocity", "RV_HARPS_leao", "source_id"]].notnull().groupby(
    ["radial_velocity", "RV_HARPS_leao"]
).agg("count")

# %%
delta_rv = df["radial_velocity"] - df["RV_HARPS_leao"]
delta_rv_sigma = delta_rv / np.hypot(df["radial_velocity_error"], df["eRV_HARPS_leao"])

mean_delta_rv = np.nanmean(delta_rv)
mean_delta_rv_sigma = np.nanmean(delta_rv_sigma)
print(f"mean delta RV (DR2-HARPS) = {mean_delta_rv:-8.4f}")
print(f"mean delta RV (DR2-HARPS) / error = {mean_delta_rv_sigma:-8.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1 = sns.distplot(
    delta_rv[~np.isnan(delta_rv)],
    ax=ax1,
    color="k",
    hist_kws={"lw":0},
    kde_kws={"lw": 1},
)
ax1.axvline(0, c="k", lw=1)
ax1.set_xlabel(r"$\mathrm{RV}_\mathrm{DR2} - \mathrm{RV}_\mathrm{HARPS}$")
ax1.set_ylabel("Density")
ax1.text(
    0.05,
    0.95,
    f"mean={mean_delta_rv:-.3f} km/s",
    ha="left",
    va="top",
    size=12,
    transform=ax1.transAxes,
)
ax1.set_yticks([0, .5, 1, 1.5, 2.])
ax1.set_ylim(0, 2.2)

sns.distplot(
    delta_rv_sigma[~np.isnan(delta_rv_sigma)],
    ax=ax2,
    color="k",
    hist_kws={"lw":0},
    kde_kws={"lw": 1},
)
ax2.axvline(0, c="k", lw=1)
ax2.set_xlabel(
    r"$\mathrm{RV}_\mathrm{DR2} - \mathrm{RV}_\mathrm{HARPS}"
    r"/ \sqrt{\sigma_\mathrm{RV, DR2}^2+\sigma_\mathrm{RV, HARPS}^2}$"
)
ax2.set_ylabel("Density")
fig.tight_layout()
fig.savefig("../plots/compare-gaia-harps-rv.pdf")

# %%
mean_cartv_icrs = [-6.03, 45.56, 5.57]
vx, vy, vz = mean_cartv_icrs

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for cax in ax:
    cax.set_aspect("equal")
for dset, color in zip([df, gdr2], ["k", None]):
    cartx, cartv = dset.g.icrs.cartesian, dset.g.icrs.velocity
    dvx, dvy, dvz = cartv.d_xyz.value - np.array(mean_cartv_icrs)[:,None]
    cond = (np.abs(dvx)<5) & (np.abs(dvy)<5) & (np.abs(dvz)<5)
#     ax[0].scatter(cartx.x, cartx.y, s=1, c=color)
    ax[0].quiver(cartx.x[cond], cartx.y[cond], dvx[cond], dvy[cond], color=color)
    ax[1].quiver(cartx.x[cond], cartx.z[cond], dvx[cond], dvz[cond], color=color)
for cax in ax:
    cax.set_xlabel("$X_\mathrm{ICRS}$")
ax[0].set_ylabel("$Y_\mathrm{ICRS}$")
ax[1].set_ylabel("$Z_\mathrm{ICRS}$")
fig.tight_layout()
fig.savefig('../plots/hyades-xyz-vector-icrs.pdf')

# %%
mean_cartv_galactic = [-42.24, -19.00, -1.48]
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for cax in ax:
    cax.set_aspect("equal")
for dset, color in zip([df, gdr2], ["k", None]):
    cartx, cartv = dset.g.galactic.cartesian, dset.g.galactic.velocity
    dvx, dvy, dvz = cartv.d_xyz.value - np.array(mean_cartv_galactic)[:, None]
    cond = (np.abs(dvx) < 3) & (np.abs(dvy) < 3) & (np.abs(dvz) < 3)
    #     ax[0].scatter(cartx.x, cartx.y, s=1, c=color)
    ax[0].quiver(cartx.x[cond], cartx.y[cond], dvx[cond], dvy[cond], color=color)
    ax[1].quiver(cartx.x[cond], cartx.z[cond], dvx[cond], dvz[cond], color=color)
for cax in ax:
    cax.set_xlabel("$X_\mathrm{Galactic}$")
ax[0].set_ylabel("$Y_\mathrm{Galactic}$")
ax[1].set_ylabel("$Z_\mathrm{Galactic}$")
fig.tight_layout()
fig.savefig('../plots/hyades-xyz-vector-galactic.pdf')

# %%
mean_cartv_galactic = [-42.24, -19.00, -1.48]
fig, ax = plt.subplots(
    3, 3, figsize=(6.5, 6.5), sharex="col", sharey="all"
)

dset = df
cartx, cartv = dset.g.galactic.cartesian, dset.g.galactic.velocity
dvx, dvy, dvz = cartv.d_xyz.value - np.array(mean_cartv_galactic)[:, None]

xyz = cartx.xyz.value
dvxyz = [dvx, dvy, dvz]

for icol in range(3):
    for irow in range(3):
        ax[irow, icol].scatter(xyz[icol], dvxyz[irow], s=1)

ax[0, 0].set_ylim(-5, 5)
for cax in ax.ravel():
    cax.set_yticks([-4, -2, 0, 2, 4])
    cax.tick_params(width=1, length=6)
fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.15, bottom=0.15, top=0.94)
for cax, label in zip(ax[:, 0], ["x", "y", "z"]):
    cax.set_ylabel(
        r"$\Delta v_{0}$".format(label) + r" [$\mathrm{km}\,\mathrm{s}^{-1}$]"
    )
ax[2, 0].set_xlabel("$X$ [pc]")
ax[2, 1].set_xlabel("$Y$ [pc]")
ax[2, 2].set_xlabel("$Z$ [pc]")
fig.suptitle(
    "Residual velocities vs. position (Galactic) $N$={}/{}".format(
        (~np.isnan(dvx)).sum(), len(df)
    ), size=15
)
fig.subplots_adjust(right=0.98, left=0.1, bottom=0.1)
fig.savefig("../plots/residual-velocity-vs-position-galactic.pdf")

# %%
error_summary = pd.DataFrame(
    dict(
        pmra_error_frac=np.abs(df["pmra_error"] / df["pmra"]),
        pmdec_error_frac=np.abs(df["pmdec_error"] / df["pmdec"]),
        parallax_error_frac=np.abs(df["parallax_error"] / df["parallax"]),
    )
).describe()
error_summary

# %%

pmdelta = np.hypot( *(df_gfr[['pmra', 'pmdec']].values - df[['pmra', 'pmdec']].values).T)
plt.scatter(df['phot_g_mean_mag'],  pmdelta, s=4);
plt.xlabel('$G$ [mag]')
plt.ylabel(r'$\Delta \mu$');

# %%
deltav = np.hypot((df_gfr.g.vra-df.g.vra).values, (df_gfr.g.vdec-df.g.vdec).values)
plt.scatter(df['phot_g_mean_mag'],  deltav, s=4);
plt.xlabel('$G$ [mag]')
plt.ylabel(r'$\Delta v_{\mathrm{tan}}$');

# %%
mean_cartv_icrs = [-6.03, 45.56, 5.57]
fig, ax = plt.subplots(
    3, 3, figsize=(6.5, 6.5), sharex="col", sharey="all"
)

dset = df
cartx, cartv = dset.g.icrs.cartesian, dset.g.icrs.velocity
dvx, dvy, dvz = cartv.d_xyz.value - np.array(mean_cartv_icrs)[:, None]

xyz = cartx.xyz.value
dvxyz = [dvx, dvy, dvz]

for icol in range(3):
    for irow in range(3):
        ax[irow, icol].scatter(xyz[icol], dvxyz[irow], s=1)

ax[0, 0].set_ylim(-5, 5)
for cax in ax.ravel():
    cax.set_yticks([-4, -2, 0, 2, 4])
    cax.tick_params(width=1, length=6)
fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.15, bottom=0.15, top=0.85)
for cax, label in zip(ax[:, 0], ["x", "y", "z"]):
    cax.set_ylabel(r"$\Delta v_{0}$".format(label)+r" [$\mathrm{km}\,\mathrm{s}^{-1}$]")
ax[2,0].set_xlabel("$X$ [pc]")
ax[2,1].set_xlabel("$Y$ [pc]")
ax[2,2].set_xlabel("$Z$ [pc]")
fig.suptitle(
    "Residual velocities vs. position (ICRS) $N$={}/{}".format(
        (~np.isnan(dvx)).sum(), len(df)
    ), size=15
)
fig.subplots_adjust(right=0.98, left=0.1, bottom=0.1, top=0.94)
fig.savefig("../plots/residual-velocity-vs-position-icrs.pdf")

# %%
fig, ax = plt.subplots(1, 1)
ax.set_xlabel("$G$ [mag]")
n_bright_sources = (df["phot_g_mean_mag"] < 12).sum()
print(n_bright_sources)

ax.hist(
    df["phot_g_mean_mag"],
    bins=np.linspace(0, 20, 21),
    histtype="step",
    color="k",
    label="all (N={})".format(len(df)),
)
ax.hist(
    df.dropna(subset=["radial_velocity"])["phot_g_mean_mag"],
    bins=np.linspace(0, 20, 21),
    histtype="step",
    label="has Gaia RV (N={})".format(df["radial_velocity"].notna().sum()),
)
ax.hist(
    df.dropna(subset=["RV_HARPS_leao"])["phot_g_mean_mag"],
    bins=np.linspace(0, 20, 21),
    histtype="step",
    label="has HARPS RV (N={})".format(df["RV_HARPS_leao"].notna().sum()),
)
ax.legend(loc="upper left", fontsize=10, frameon=False);
ax.set_ylabel('Count');

# %%
df = out_full.loc[out_full["Member_r19"] != "other"]
fig, ax = plt.subplots()
ax.scatter(
    df["bp_rp"],
    df["phot_g_mean_mag"] + df.g.distmod,
    s=1, c='k'
)


ax.invert_yaxis()
ax.set_xlabel("BP-RP [mag]")
ax.set_ylabel("$M_G$ [mag]");

# %%
# get tgas data for velocity uncertainty comparison
hy_tgas = pd.read_csv("../data/reino_tgas_full.csv", index_col=0)
print(f"number of sources in Reino selection: {len(hy_tgas)} rows")

tmp = pd.concat(
    [
        hy_tgas.g.vra_error.rename("v").to_frame().assign(label=r"TGAS $v_\alpha$"),
        hy_tgas.g.vdec_error.rename("v").to_frame().assign(label=r"TGAS $v_\delta$"),
        df.g.vra_error.rename("v").to_frame().assign(label=r"DR2 $v_\alpha$"),
        df.g.vdec_error.rename("v").to_frame().assign(label=r"DR2 $v_\delta$"),
        #         df.g.vra_error.rename('v').to_frame().assign(label='HG vra'),
        #         df.g.vdec_error.rename('v').to_frame().assign(label='HG vdec'),
        df["radial_velocity_error"].rename("v").to_frame().assign(label="DR2 RV"),
        df["eRV_HARPS_leao"].rename("v").to_frame().assign(label="HARPS RV"),
    ]
)
tmp["v"] = np.log10(tmp["v"])
tmp.groupby('label').describe()

g = sns.FacetGrid(tmp, row="label", aspect=5, height=0.8)
g.map(sns.kdeplot, "v", clip_on=False, shade=True, alpha=1, lw=1.5, bw=0.2)
g.set_titles("")
g.fig.subplots_adjust(hspace=0.1, top=0.95, right=0.95, left=0.05, bottom=0.12)

g.set(xticklabels=["0.001", "0.01", "0.1", "1", "10"], xticks=[-3, -2, -1, 0, 1])
g.set(yticks=[])
for cax, label in zip(g.fig.axes, g.row_names):
    cax.spines["left"].set_visible(False)
    cax.tick_params(length=5, labelsize=12)
    cax.text(0.95, 0.95, label, ha='right', va='top', transform=cax.transAxes,
             bbox=dict(facecolor='w'), size=12)
    cax.axvline(np.log10(0.3), c='k', lw=1, linestyle=':', zorder=-1);
g.fig.axes[-1].set_xlabel(r'$\log \sigma_v\,/\,[\mathrm{km}\,\mathrm{s}^{-1}$]');
g.fig.savefig("../plots/hyades-velocity-uncertainty-distribution.pdf")

# %%
