"""Fit hyades DR2 data after correcting the frame roation for bright sources"""
import os, sys
import pandas as pd
import numpy as np

import kinesis as kn
import gapipes as gp

outfile = "../report/hyades-dr2/fit_allstars_brightcorr.pickle"
if os.path.exists(outfile):
    sys.exit(f"Path {outfile} already exists; doing nothing.")
elif not os.path.exists(os.path.dirname(outfile)):
    dirname = os.path.exists(os.path.dirname(outfile))
    sys.exit(f"Output directory {dirname} does not exist; doing nothing.")

# construct data table
df = pd.read_csv("../data/hyades_full.csv")
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
# Correct bright source pm
df = df.g.correct_brightsource_pm()
data = df[necessary_columns + ["radial_velocity", "radial_velocity_error"]].copy()
b0 = np.array([17.26821532, 41.64304963, 13.606407])
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
