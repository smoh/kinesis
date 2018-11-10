from functools import wraps

import numpy as np
import pandas as pd

__all__ = ['cache_to', 'cov_from_gaia_table']


def cache_to(path):
    """
    Decorator to cache pandas DataFrame to csv
    """
    def decorator_cache(func):
        @wraps(func)
        def wrapper_cache():
            try:
                r = pd.read_csv(path)
                print("Data loaded from {:s}".format(path))
                return r
            except FileNotFoundError:
                r = func()
                r.to_csv(path)
                print("Data written to {:s}".format(path))
                return r
        return wrapper_cache
    return decorator_cache


def cov_from_gaia_table(df):
    """
    Returns array of covariance matrix of parallax, pmra, pmdec
    with shape (N, 3, 3).
    """
    necessary_columns = set([
        'ra', 'dec', 'parallax', 'pmra', 'pmdec',
        'parallax_error', 'pmra_error', 'pmdec_error',
        'parallax_pmra_corr', 'parallax_pmdec_corr',
        'pmra_pmdec_corr'])
    s = set(df.columns)
    assert s > necessary_columns, \
        "Columns missing: {:}".format(necessary_columns-s)
    C = np.zeros([len(df), 3, 3])
    C[:, [0,1,2], [0,1,2]] = df[['parallax_error', 'pmra_error', 'pmdec_error']].values**2
    C[:, [0, 1], [1, 0]] = (df['parallax_error']*df['pmra_error']*df['parallax_pmra_corr']).values[:, None]
    C[:, [0, 2], [2, 0]] = (df['parallax_error']*df['pmdec_error']*df['parallax_pmdec_corr']).values[:, None]
    C[:, [1, 2], [2, 1]] = (df['pmra_error']*df['pmdec_error']*df['pmra_pmdec_corr']).values[:, None]
    return C
