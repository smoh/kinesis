import numpy as np
import pandas as pd

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord

DATADIR = '/Users/semyeong/projects/kinesis/data/'


__all__ = ['load_reino2018_supp', 'load_reino2018_full']


def load_reino2018_supp():
     return pd.read_csv(
        DATADIR+'Reino2018_supp.txt', comment='#', delim_whitespace=True,
        dtype={'HIP': str, 'TYC': str, 'source_id': str})


def load_reino2018_full():
    return pd.read_csv(DATADIR+"reino_tgas_full.csv")
