import numpy as np
import pandas as pd

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord

import gapipes

DATADIR = "/Users/semyeong/projects/kinesis/data/"


__all__ = ["load_reino2018_supp", "load_reino2018_full", "load_hyades_dataset"]


def load_reino2018_supp():
    return pd.read_csv(
        DATADIR + "Reino2018_supp.txt",
        comment="#",
        delim_whitespace=True,
        dtype={"HIP": str, "TYC": str, "source_id": str},
    )


def load_reino2018_full():
    return pd.read_csv(DATADIR + "reino_tgas_full.csv")


def load_hyades_dataset():
    df = pd.read_csv("../data/hyades_full.csv")
    df["x"], df["y"], df["z"] = df.g.icrs.cartesian.xyz.value
    df["vx"], df["vy"], df["vz"] = df.g.icrs.velocity.d_xyz.value
    df["gx"], df["gy"], df["gz"] = df.g.galactic.cartesian.xyz.value
    df["gvx"], df["gvy"], df["gvz"] = df.g.galactic.velocity.d_xyz.value
    return df
