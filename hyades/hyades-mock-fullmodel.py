#!/usr/bin/env python
# coding: utf-8

# Generate mock Hyades data with DR2 uncertainties and fit with the full model.

import os, sys
import pandas as pd
import numpy as np
import click

# project
import pystan
import gapipes as gp
import kinesis as kn

print(f"pystan v{pystan.__version__}")

# hard-coded center value
b_c_icrs = np.array([17.15474298, 41.28962638, 13.69105771])


def check_path(outfile):
    """Strictly check given output file path"""
    if os.path.exists(outfile):
        sys.exit(f"Path {outfile} already exists; doing nothing.")
    elif not os.path.exists(os.path.dirname(outfile)):
        dirname = os.path.exists(os.path.dirname(outfile))
        sys.exit(f"Output directory {dirname} does not exist; doing nothing.")


@click.command()
@click.argument("outfile")
@click.option("--seed", type=int, help="random number generator seed")
@click.option("--dryrun", is_flag=True, help="do not actually do fitting")
@click.option("--use_harps_rv", is_flag=True, help="swap DR2 RVs with HARPS RVs")
def main(outfile, seed=382948, dryrun=False, use_harps_rv=False):
    """
    Generate and fit mock data
    """
    check_path(outfile)

    df0 = pd.read_csv("../data/hyades_full.csv")
    df0 = df0.loc[df0["Member_r19"] != "other"].copy().reset_index(drop=True)
    r1, r2 = 0, 10
    xyz = df0.g.icrs.cartesian.xyz.value
    r_c = np.linalg.norm(xyz - b_c_icrs[:, None], axis=0)
    df = df0.loc[(r_c > r1) & (r_c < r2)].copy()

    if use_harps_rv:
        df['radial_velocity'] = df['RV_HARPS_leao']
        df['radial_velocity_error'] = df['eRV_HARPS_leao']

    # Randomly divide into 90% member and 10% background
    N = len(df)
    idx_mem = np.sort(np.random.choice(np.arange(len(df)), size=380, replace=False))
    idx_bg = np.sort(np.array(list(set(np.arange(len(df))) - set(idx_mem))))
    truth_T0 = dict(
        b0=[17.7, 41.2, 13.3], v0=np.array([-6.32, 45.24, 5.30]), sigmav=0.3
    )

    cl = (
        kn.Cluster(**truth_T0)
        .sample_at(df.iloc[idx_mem].g.icrs)
        .observe(
            cov=df.iloc[idx_mem].g.make_cov(),
            rv_error=df.iloc[idx_mem]["radial_velocity_error"],
            # rv_error=df.iloc[idx_mem]["radial_velocity_error"].fillna(1).values,
            # rv_error=np.array([0.05] * len(idx_mem)),
        )
    )
    data_cl = cl.members.observed.copy()
    print(
        "apparent sigma [x, y, z] =",
        np.nanstd(data_cl.g.icrs.velocity.d_xyz.value, axis=1),
    )

    bg = (
        kn.Cluster(b0=[17.7, 41.2, 13.3], v0=[0, 0, 0], sigmav=20)
        .sample_at(df.iloc[idx_bg].g.icrs)
        .observe(
            cov=df.iloc[idx_bg].g.make_cov(),
            rv_error=df.iloc[idx_bg]["radial_velocity_error"],
            # rv_error=df.iloc[idx_bg]["radial_velocity_error"].fillna(1).values,
            # rv_error=np.array([0.05] * len(idx_bg)),
        )
    )
    data_bg = bg.members.observed.copy()
    data = data_cl.assign(mem=1).append(data_bg.assign(mem=0)).reset_index(drop=True)

    stanmodel = kn.get_model("allcombined", recompile=False)
    # build input data to stan
    b0 = truth_T0["b0"]
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

    if dryrun:
        return
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


if __name__ == "__main__":
    main()
