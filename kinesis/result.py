"""Utilities to summarize fit result
"""
import pandas as pd
import arviz as az

__all__ = ["compare_param"]


def compare_param(azfit_dict, var_names, **kwargs):
    """Compare parameter summary for multiple fits

    Returns MuliIndex DataFrame
    """
    dfs = [az.summary(cfit, var_names) for k, cfit in azfit_dict.items()]
    return pd.concat(dfs, keys=azfit_dict.keys())
