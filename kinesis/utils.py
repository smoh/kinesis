import pickle
import numpy as np

__all__ = [
    "save_stanfit",
    "load_stanfit",
    "cov_from_gaia_table",
]


def save_stanfit(stanfit, outfile):
    """Save stanfit object as pickle

    stanfit (pystan.StanFit4Model): fit object
    outfile (str): path to output file
    """
    model = stanfit.stanmodel
    with open(outfile, "wb") as f:
        pickle.dump({"model": model, "fit": stanfit}, f, protocol=-1)


def load_stanfit(filename):
    """Load stanfit object from a pickle file"""
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)
    fit = data_dict["fit"]
    return fit


def cov_from_gaia_table(df):
    """
    Returns array of covariance matrix of parallax, pmra, pmdec
    with shape (N, 3, 3).
    """
    necessary_columns = set(
        [
            "ra",
            "dec",
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
    )
    s = set(df.columns)
    assert s >= necessary_columns, "Columns missing: {:}".format(necessary_columns - s)
    C = np.zeros([len(df), 3, 3])
    C[:, [0, 1, 2], [0, 1, 2]] = (
        df[["parallax_error", "pmra_error", "pmdec_error"]].values ** 2
    )
    C[:, [0, 1], [1, 0]] = (
        df["parallax_error"] * df["pmra_error"] * df["parallax_pmra_corr"]
    ).values[:, None]
    C[:, [0, 2], [2, 0]] = (
        df["parallax_error"] * df["pmdec_error"] * df["parallax_pmdec_corr"]
    ).values[:, None]
    C[:, [1, 2], [2, 1]] = (
        df["pmra_error"] * df["pmdec_error"] * df["pmra_pmdec_corr"]
    ).values[:, None]
    return C
