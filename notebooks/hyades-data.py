# -*- coding: utf-8 -*-
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
plt.style.use("smoh")


# %% [markdown]
# We consider
# - astrometry:
#     * Gaia DR2
#     * Hipparcos-Gaia
# - radial velocity:
#     * Gaia DR2
#     * HARPS
# - membership:
#     * Léao 2019 = Lindegren 2000 + outlier rejection
#     * Gaia collab.
#     * Röser 2019 (tails)
#     * Meingast 2019 (tails)

# %% {"tags": ["data"]}
def get_hyades_dr2_full():
    # Gaia Collaboration DR2 selection
    datadir = "../data/gaia_dr2_clusters"
    tmp1 = Table.read(
        f"{datadir}/tablea1a.dat", format="ascii.cds", readme=f"{datadir}/ReadMe"
    ).to_pandas()
    tmp1.columns = tmp1.columns.str.lower()
    tmp1 = tmp1.rename(columns={"source": "source_id"})
    tmp2 = pd.read_csv(f"{datadir}/table1a_gaiadr2_full.csv", index_col=0)
    dr2cl = pd.merge(tmp1[["source_id", "cluster"]], tmp2, on="source_id")

    hy_dr2 = dr2cl.groupby("cluster").get_group("Hyades")
    return hy_dr2


gdr2 = get_hyades_dr2_full()[["source_id"]]
gdr2["in_dr2"] = True
print(f"number of sources in Gaia DR2 selection: {len(gdr2)} rows")

# %%
leao = pd.read_csv("../data/leao_table.csv", sep=" ", skipinitialspace=True)
print(f"{len(leao)} rows")
hip_harps = set(leao["HIP"].values)

query_leao_sourceid = """
-- cross-match HIPPARCOS2 sources by id
SELECT
  t.hip, xm.angular_distance, xm.number_of_neighbours, gaia.source_id
FROM TAP_UPLOAD.t AS t
JOIN gaiadr2.hipparcos2_best_neighbour AS xm
  ON xm.original_ext_source_id = t.hip
JOIN gaiadr2.gaia_source AS gaia
  ON xm.source_id = gaia.source_id
"""
leaox = gp.gaia.query(query_leao_sourceid, upload_table_name="t", upload_resource=t[["ID", "HIP"]])[
    ["source_id", "hip"]
]
leaox["in_leao"] = True

missing_in_gdr2 = hip_harps - set(leaox["hip"])
print(f"leao sources missing in gdr2: {len(missing_in_gdr2)}")
print(f"missing HIP: {missing_in_gdr2}")

# %%
hy_tails_m19 = Table.read(
    "/home/soh/data/meingast2019_hyades_tails/hyades.dat",
    format="ascii.cds",
    readme="/home/soh/data/meingast2019_hyades_tails/ReadMe",
).to_pandas()
m19 = hy_tails_m19.rename(columns={"Source": "source_id"})[["source_id"]]
m19["source_id"] = m19["source_id"].astype(int)
m19["in_meingast"] = True
print(f"Meingast N={len(m19):4d}")

# %%
hy_tails_r19 = Table.read(
    "/home/soh/data/roeser_hyades_tails/stars.dat",
    format="ascii.cds",
    readme="/home/soh/data/roeser_hyades_tails/ReadMe",
).to_pandas()
r19 = hy_tails_r19.rename(columns={"Source": "source_id"})[["source_id"]]
r19["source_id"] = r19["source_id"].astype(int)
r19["in_roser"] = True
print(f"Roeser   N={len(r19):4d}")


# %%
from functools import reduce

dfs = [gdr2, leaox, m19, r19]
out = reduce(lambda left, right: pd.merge(left, right, how="outer"), dfs)


# %%
# finally, query DR2 astrometry with source_id
out_full = gp.gaia.query_sourceid(out[['source_id']], columns='gaiadr2.gaia_source.*')
out_full = out_full.merge(out, on='source_id')

# %%
out_full = out_full.merge(
    hy_tails_r19[["Source", "Member", "Comment"]].rename(
        columns={
            "Source": "source_id",
            "Member": "Member_r19",
            "Comment": "Comment_r19",
        }
    ),
    how="left",
    on="source_id",
)
out_full = out_full.merge(
    leao.rename(columns=lambda x: x + "_leao" if x != "HIP" else x),
    how="left",
    left_on="hip",
    right_on="HIP",
).drop(columns='HIP')

# %%
assert out_full['source_id'].duplicated().sum() == 0
print(f"{len(out_full)} rows in the combined catalog")

# %%
# %store out_full

# %%
print(out_full[['in_dr2', 'in_leao', 'in_meingast', 'in_roser']].count())
