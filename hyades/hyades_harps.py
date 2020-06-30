"""Cross-match Leao HARPS RVs with Gaia DR2 and HGCA
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
from gapipes.gaia import gaia
from astropy.table import Table

#%%
t = pd.read_csv("data/leao_table.csv", sep=" ", skipinitialspace=True)
hip_harps = set(t["HIP"].values)

#%% Query GaiaArchive for cross-matched astrometry

q = """
-- cross-match HIPPARCOS2 sources by id
SELECT
  t.hip, gaia.*, xm.angular_distance, xm.number_of_neighbours
FROM TAP_UPLOAD.t AS t
JOIN gaiadr2.hipparcos2_best_neighbour AS xm
  ON xm.original_ext_source_id = t.hip
JOIN gaiadr2.gaia_source AS gaia
  ON xm.source_id = gaia.source_id
"""
xga = gaia.query(q, upload_table_name="t", upload_resource=t[["ID", "HIP"]])
hip_missing_xga = hip_harps - set(xga["hip"])
n_missing_xga = len(hip_missing_xga)
print("{:d} sources missing in GaiaArchive".format(n_missing_xga))
print("HIP missing in GaiaArchive", hip_missing_xga)

#%% Query HGCA for cross-matched astrometry
hg = Table.read("data/HGCA_vDR2_corrected.fits").to_pandas()

xhg = pd.merge(t[["HIP"]], hg, left_on="HIP", right_on="hip_id")
hip_missing_xhg = hip_harps - set(xhg["HIP"])
n_missing_xhg = len(hip_missing_xhg)
print("{:d} sources missing in HGCA".format(n_missing_xhg))
print("HIP missing in HGCA:", hip_missing_xhg)

#%% Check whether matched sources agree between GaiaArchive and HGCA
x_ga_hg = pd.merge(
    xga[["hip", "source_id"]],
    xhg[["HIP", "gaia_source_id"]],
    left_on="hip",
    right_on="HIP",
)
if (x_ga_hg["gaia_source_id"].values == x_ga_hg["source_id"].values).all():
    print("For HIP in common between Gaia and HGCA, all matched source_ids agree.")

#%% Merge, clean and save Gaia and HGCA cross-matched tables

t.columns = t.columns.str.lower()
xga = pd.merge(t, xga, on="hip", suffixes=("_leao", ""))

xhg.columns = xhg.columns.str.lower()
xhg = xhg.drop(columns="hip_id")  # duplicated with HIP
xhg = pd.merge(t, xhg, on="hip")

xga.to_csv("data/hyades_gaia_harps.csv")
xhg.to_csv("data/hyades_hgca_harps.csv")
