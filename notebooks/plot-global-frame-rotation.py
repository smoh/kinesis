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

# %% {"tags": ["setup"]}
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

plt.style.use("smoh")

# %%
# %%time
r = gp.gaia.query(
    """
-- From Hipparcos-Gaia cross-match, get positions and
-- proper motion "residual" between Gaia-measured proper motion and
-- proper motion calculated as positional difference between two epochs.
SELECT
    gaia.ra, gaia.dec,
    gaia.pmra - (gaia.ra - hip.ra) *3600000 * COS(RADIANS(gaia.dec)) / 24.25 AS "delta_pmra",
    gaia.pmdec - (gaia.dec - hip.dec) *3600000/ 24.25 as "delta_pmdec"
FROM gaiadr2.gaia_source gaia
    JOIN gaiadr2.hipparcos2_best_neighbour hip_match
    ON gaia.source_id = hip_match.source_id
    JOIN public.hipparcos_newreduction hip
    ON hip_match.original_ext_source_id = hip.hip
"""
)
print(f"{len(r)} rows")

# %%
from scipy.stats import gaussian_kde, binned_statistic_2d
import scipy.ndimage as ndimage


# %%
def plot_binned_statistic_2d(
    x,
    y,
    values,
    statistic="mean",
    bins=10,
    range=None,
    gaussian_filter=None,
    ax=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    s, xe, ye, bn = binned_statistic_2d(
        x, y, values, statistic=statistic, bins=bins, range=range
    )
    if gaussian_filter is not None:
        s1 = s.copy()
        s1[np.isnan(s1)] = 0
        img1 = ndimage.gaussian_filter(s1, **gaussian_filter)

        s2 = s.copy() * 0.0 + 1.0
        s2[np.isnan(s)] = 0.0
        img2 = ndimage.gaussian_filter(s2, **gaussian_filter)

        img = img1 / img2
    else:
        img = s
    x, y = np.meshgrid(xe, ye)
    ax.pcolormesh(x, y, img.T, **kwargs)
    return ax


# %%
# %store -r out_full
d = out_full.loc[out_full['Member_r19']!='other']

# %%
fig, ax = plt.subplots(2, 1, subplot_kw=dict(projection=ccrs.Mollweide()))
plot_binned_statistic_2d(
    r["ra"],
    r["dec"],
    r["delta_pmra"],
    gaussian_filter={"sigma": 8, "mode": "wrap"},
    bins=(361, 181),
    ax=ax[0],
    transform=ccrs.RotatedPole(),
    vmin=-0.2,
    vmax=0.2,
    cmap="RdYlBu",
)
plot_binned_statistic_2d(
    r["ra"],
    r["dec"],
    r["delta_pmdec"],
    gaussian_filter={"sigma": 3, "mode": "wrap"},
    bins=(181, 91),
    ax=ax[1],
    transform=ccrs.RotatedPole(),
    vmin=-0.2,
    vmax=0.2,
    cmap="RdYlBu",
)
for cax in ax:
    cax.scatter(d['ra'], d['dec'], transform=ccrs.Geodetic(), s=1, c='k')
    cax.invert_xaxis()
    # first of the collections is pcolormesh of the background
    cb = plt.colorbar(cax.collections[0], ax=cax)
    cb.ax.tick_params(width=1, length=5)

ax[0].set_title(r'$\Delta$pmra')
ax[1].set_title(r'$\Delta$pmdec')
fig.tight_layout()
fig.savefig("../plots/hip-gaia-residual-proper-motion.pdf");
# %%
# %%time
deltapm_healpix = gp.gaia.query(
    """
-- From Hipparcos-Gaia cross-match, get average positions and
-- proper motion "residual" grouped by healpix pixels.
SELECT
    avg(gaia.ra) as "avg_ra", avg(gaia.dec) as "avg_dec",
    gaia_healpix_index(4, gaia.source_id) AS healpix_6,
    avg(gaia.pmra - (gaia.ra - hip.ra) *3600000 * COS(RADIANS(gaia.dec)) / 24.25) as "avg_delta_pmra",
    avg(gaia.pmdec - (gaia.dec - hip.dec) *3600000/ 24.25) as "avg_delta_pmdec",
    count(gaia.source_id)
FROM gaiadr2.gaia_source gaia
    JOIN gaiadr2.hipparcos2_best_neighbour hip_match
    ON gaia.source_id = hip_match.source_id
    JOIN public.hipparcos_newreduction hip
    ON hip_match.original_ext_source_id = hip.hip
GROUP BY healpix_6
ORDER BY healpix_6
"""
)

# %%
# there should be no missing healpix pixels for plotting
npix = hp.nside2npix(2**4)
missing_ipix = sorted(list(set(range(npix)) - set(deltapm_healpix['healpix_6'].values)))
print("number of missing pixels =", len(missing_ipix))
print('number of sources per pixel')
print(deltapm_healpix['COUNT'].quantile([.05, .5, .95]))

import healpy as hp

fig = plt.figure(figsize=(4, 4))
hp.mollview(
    deltapm_healpix["avg_delta_pmdec"],
    nest=True,
    min=-0.3,
    max=0.3,
    cmap="RdYlBu",
    fig=fig.number,
    title='avg_delta_pmdec'
)
fig = plt.figure(figsize=(4, 4))
hp.mollview(
    deltapm_healpix["avg_delta_pmra"],
    nest=True,
    min=-0.3,
    max=0.3,
    cmap="RdYlBu",
    fig=fig.number,
    title='avg_delta_pmra'
)
