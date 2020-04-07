import pickle
from functools import wraps

import numpy as np
import pandas as pd
import astropy.coordinates as coord

__all__ = [
    "save_stanfit",
    "load_stanfit",
    "cov_from_gaia_table",
    "decompose_T",
    "rotate_T_to_galactic",
    "EigenvalueDecomposition",
    "add_transformed_posterior",
    "add_cartesian_xv",
    "reconstruct_df_from_stanfit",
]


def save_stanfit(stanfit, outfile):
    """Save stanfit object as pickle

    stanfit : pystan.StanFit4Model
        fit object
    outfile : str

        path to output file
    """
    model = stanfit.stanmodel
    with open(outfile, "wb") as f:
        pickle.dump({"model": model, "fit": stanfit}, f, protocol=-1)


def load_stanfit(filename):
    """Load stanfit object from a pickle file
    """
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)
    fit = data_dict["fit"]
    return fit


def decompose_T(T):
    """Decompose velocity gradient tensor to interpretable components

    T : array, (..., 3, 3)
        dv_j / dv_i
    
    Returns
    -------
    dict

    
    """
    if T.shape[-2:] != (3, 3):
        raise ValueError("`T` must have shape (..., 3, 3)")
    omegax = 0.5 * (T[..., 2, 1] - T[..., 1, 2])
    omegay = 0.5 * (T[..., 0, 2] - T[..., 2, 0])
    omegaz = 0.5 * (T[..., 1, 0] - T[..., 0, 1])

    w1 = 0.5 * (T[..., 2, 1] + T[..., 1, 2])
    w2 = 0.5 * (T[..., 0, 2] + T[..., 2, 0])
    w3 = 0.5 * (T[..., 1, 0] + T[..., 0, 1])
    w4 = T[..., 0, 0]
    w5 = T[..., 1, 1]
    kappa = (w4 + w5 + T[..., 2, 2]) / 3.0
    T = T.squeeze()
    return dict(
        omegax=omegax,
        omegay=omegay,
        omegaz=omegaz,
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
        w5=w5,
        kappa=kappa,
    )


def rotate_T_to_galactic(T):
    """R T R^T"""
    # hack rotation matrix from icrs -> galactic
    rotmat = (
        coord.ICRS(
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            representation_type=coord.CartesianRepresentation,
        )
        .transform_to(coord.Galactic)
        .cartesian.xyz.value
    )
    rotated_T = np.einsum("ij,...jk,kl->...il", rotmat, T, rotmat.T)
    return rotated_T


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


class EigenvalueDecomposition(object):
    def __init__(self, a):
        """
        a : (N, 3, 3)
        """
        self.a = a
        w, v = np.linalg.eig(a)
        i_wsort = np.argsort(w, axis=1)
        sorted_v = np.stack([v[j][:, i] for j, i in enumerate(i_wsort)])
        sorted_w = np.sort(w, axis=1)
        self.w = sorted_w
        self.v = sorted_v

        # eigenvectors to angles
        theta = np.rad2deg(np.arctan(sorted_v[:, 1, :] / sorted_v[:, 0, :]))
        R = np.hypot(sorted_v[:, 0, :], sorted_v[:, 1, :])
        cosphi = np.cos(np.arctan(sorted_v[:, 2, :] / R))
        self.theta = theta
        self.cosphi = cosphi


def make_summary_dataframe(azfit):
    pass


def add_transformed_posterior(azfit):
    """Add transformed posterior samples to az.InferenceData

    Returns az.InferenceData with transformed samples added

    Added parameters:
    Sigma : (3, 3) Dispersion matrix
    omegax, omegay, omegaz, w1, w2, w3, w4, w5, kappa:
        decomposed paramters of velocity gradient tensor
    
    *_gal : above parameters in Galactic frame
    """
    v = azfit

    for ck, cv in decompose_T(v.posterior["T_param"]).items():
        v.posterior[ck] = cv
    # Combine scale and correlation matrix of Sigma to variance matrix
    sigv_samples, Omega_samples = v.posterior["sigv"], v.posterior["Omega"]
    Sigma_samples = np.einsum(
        "cni,cnij,cnj->cnij", sigv_samples, Omega_samples, sigv_samples
    )
    v.posterior["Sigma"] = (
        ("chain", "draw", "Sigma_dim_0", "Sigma_dim_1"),
        Sigma_samples,
    )
    v.posterior["Sigma_gal"] = (
        ("chain", "draw", "Sigma_dim_0", "Sigma_dim_1"),
        rotate_T_to_galactic(Sigma_samples),
    )
    # Add rotated T matrix and decomposition
    v.posterior["T_param_gal"] = (
        ("chain", "draw", "dim0", "dim1"),
        rotate_T_to_galactic(v.posterior["T_param"]),
    )
    for ck, cv in decompose_T(v.posterior["T_param_gal"]).items():
        v.posterior[ck + "_gal"] = cv
    return v


def add_cartesian_xv(df):
    df["x"], df["y"], df["z"] = df.g.icrs.cartesian.xyz.value
    df["vx"], df["vy"], df["vz"] = df.g.icrs.velocity.d_xyz.value
    df["gx"], df["gy"], df["gz"] = df.g.galactic.cartesian.xyz.value
    df["gvx"], df["gvy"], df["gvz"] = df.g.galactic.velocity.d_xyz.value


# add mean probmem to data
def reconstruct_df_from_stanfit(stanfit):
    dat = stanfit.data
    reconstructed_df = pd.DataFrame(
        dict(
            parallax=dat["a"][:, 0],
            pmra=dat["a"][:, 1],
            pmdec=dat["a"][:, 2],
            radial_velocity=np.nan,
            radial_velocity_error=np.nan,
            ra=dat["ra"],
            dec=dat["dec"],
        )
    )

    reconstructed_df["radial_velocity"].iloc[dat["irv"]] = dat["rv"]
    reconstructed_df["radial_velocity_error"].iloc[dat["irv"]] = dat["rv_error"]
    reconstructed_df["mean_pmem"] = stanfit["probmem"].mean(axis=0)
    add_cartesian_xv(reconstructed_df)
    return reconstructed_df


def add_cartesian_xv(df):
    df["x"], df["y"], df["z"] = df.g.icrs.cartesian.xyz.value
    df["vx"], df["vy"], df["vz"] = df.g.icrs.velocity.d_xyz.value
    df["gx"], df["gy"], df["gz"] = df.g.galactic.cartesian.xyz.value
    df["gvx"], df["gvy"], df["gvz"] = df.g.galactic.velocity.d_xyz.value
    return df
