#%%
"""Fit dividing at the tidal radius
"""
import os, sys
import pandas as pd
import numpy as np
import astropy.coordinates as coord

import kinesis as kn
import gapipes as gp

# #%%
# outfile = "../report/hyades-dr2/fit.pickle"
# if os.path.exists(outfile):
#     sys.exit(f"Path {outfile} already exists; doing nothing.")
# elif not os.path.exists(os.path.dirname(outfile)):
#     dirname = os.path.exists(os.path.dirname(outfile))
#     sys.exit(f"Output directory {dirname} does not exist; doing nothing.")


def fit_and_save(srcdf, outfile):
    necessary_columns = [
        "ra",
        "dec",
        "phot_g_mean_mag",
        "parallax",
        "pmra",
        "pmdec",
        "parallax_error",
        "pmra_error",
        "pmdec_error",
        "parallax_pmra_corr",
        "parallax_pmdec_corr",
        "pmra_pmdec_corr",
    ]
    data = srcdf[
        necessary_columns + ["radial_velocity", "radial_velocity_error"]
    ].copy()
    b0 = b_c_icrs
    print(f"{len(data)} rows")
    print(f"b0 = {b0}")

    N = len(data)
    irv = np.arange(N)[data["radial_velocity"].notna()]
    rv = data["radial_velocity"].values[irv]
    rv_error = data["radial_velocity_error"].values[irv]
    data_dict = {
        "N": N,
        "Nrv": data["radial_velocity"].notna().sum(),
        "ra": data["ra"].values,
        "dec": data["dec"].values,
        "a": data[["parallax", "pmra", "pmdec"]].values,
        "C": data.g.make_cov(),
        "irv": irv,
        "rv": rv,
        "rv_error": rv_error,
        "include_T": 1,
        "b0": b0,
    }

    def stan_init():
        return dict(
            d=1e3 / data["parallax"].values,
            sigv=[0.5, 0.5, 0.5],
            Omega=np.eye(3),
            v0=[-5, 45, 5],
            T=np.zeros(shape=(1, 3, 3)),
            v0_bg=[0, 0, 0],
            sigv_bg=50.0,
            f_mem=0.95,
        )

    stanmodel = kn.get_model("allcombined")
    fit = stanmodel.sampling(
        data=data_dict,
        init=stan_init,
        pars=[
            "v0",
            "sigv",
            "Omega",
            "T_param",
            "v0_bg",
            "sigv_bg",
            "f_mem",
            "probmem",
            "a_model",
            "rv_model",
        ],
    )
    kn.save_stanfit(fit, outfile)


# %% construct data table
df = pd.read_csv("../data/hyades_full.csv")
cl_gaia = df.loc[df["in_dr2"] == True].copy()


def xyz_icrs_to_galactic(xyz):
    c = coord.ICRS(*xyz, representation_type="cartesian")
    return c.transform_to(coord.Galactic).cartesian.xyz.value


#%%

b_c_icrs = np.array([17.13924468, 41.23189102, 13.65416937])
r_cut = 9  # pc

xyz = df.g.icrs.cartesian.xyz.value
r_c = np.linalg.norm(xyz - b_c_icrs[:, None], axis=0)
df["in_r_cut"] = r_c < r_cut
df_cl = df.groupby("in_r_cut").get_group(True).copy()
df_tails = df.groupby("in_r_cut").get_group(False).copy()

fit_and_save(df_cl, "../report/hyades-dr2-rtid-9pc/cl.pickle")
fit_and_save(df_tails, "../report/hyades-dr2-rtid-9pc/tails.pickle")
