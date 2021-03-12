"""
Routines to assist analysis of fitting results
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns


__all__ = [
    "decompose_T",
    "rotate_T_to_galactic",
    "EigenvalueDecomposition",
    "add_transformed_posterior",
    "add_cartesian_xv",
    "reconstruct_df_from_stanfit",
    "plot_violins",
    "plot_mean_velocity",
    "confidence_ellipse",
    "get_flat2d",
    "compare_param",
]


def calculate_rv_residual(stanfit):
    """Calculate (rv_data - rv_model) / sqrt(rv_error^2 + sigv_model^2)

    Returns:
        res: 2d-array of (n_posterior_samples, n_rv_sources).

    Sliced in axis=0, they should be distributed as Normal(0, 1).
    """
    res = (stanfit.data["rv"][None, :] - stanfit["rv_model"]) / np.hypot(
        stanfit.data["rv_error"][None, :], stanfit["sigv"][:, None]
    )
    return res


def calculate_veca_residual(stanfit):
    """Calculate (a_data - a_model)^T * D * (a_data - a_model)

    where D is covariance matrix of observed errors + sigv.

    Returns:
        g: 2d array, (n_samples, n_sources)

    Sliced in axis=0, they should be distributed as chi2(df=3).
    """
    fit = stanfit
    n_samples = fit["sigv"].shape[0]
    delta_a = fit.data["a"][None, :] - fit["a_model"]
    D = np.repeat(fit.data["C"].copy()[None], n_samples, axis=0)
    D[:, :, 1, 1] += (fit["sigv"] ** 2)[:, None] / (fit["d"] / 1e3) ** 2 / 4.74 ** 2
    D[:, :, 2, 2] += (fit["sigv"] ** 2)[:, None] / (fit["d"] / 1e3) ** 2 / 4.74 ** 2
    Dinv = np.linalg.inv(D)
    g = np.einsum("sni,snij,snj->sn", delta_a, Dinv, delta_a)
    return g


def plot_ppc_rv(stanfit):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy as sp

    rv_res = Fitter.calculate_rv_residual(stanfit)
    for slc in rv_res:
        sns.distplot(slc, hist=False, kde_kws={"lw": 0.5})
    x = np.linspace(-5, 5, 51)
    plt.plot(x, sp.stats.norm.pdf(x), "k-")
    return plt.gcf()


def decompose_T(T):
    """Decompose velocity gradient tensor T to interpretable components

    T (array, (..., 3, 3)): dv_j / dv_i

    Returns:
        dict: decomposed T, i.e., omegax, omegay, omegaz, w1, ..., w5, and kappa.
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
    """
    Rotate T=dv_i/dv_j in ICRS to Galactic frame, R T R^T
    """
    # hack rotation matrix from icrs -> galactic
    rotmat = (
        coord.ICRS(
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            representation_type=coord.CartesianRepresentation,
        )
        .transform_to(coord.Galactic())
        .cartesian.xyz.value
    )
    rotated_T = np.einsum("ij,...jk,kl->...il", rotmat, T, rotmat.T)
    return rotated_T


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


def add_transformed_posterior(azfit):
    """Add transformed posterior samples to az.InferenceData

    Returns az.InferenceData with transformed samples added

    Added parameters:

    - Sigma : (3, 3) Dispersion matrix
    - omegax, omegay, omegaz, w1, w2, w3, w4, w5, kappa:
        decomposed paramters of velocity gradient tensor
    - *_gal : above parameters in Galactic frame
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

    # add mean probmem to data


def add_cartesian_xv(df):
    df["x"], df["y"], df["z"] = df.g.icrs.cartesian.xyz.value
    df["vx"], df["vy"], df["vz"] = df.g.icrs.velocity.d_xyz.value
    df["gx"], df["gy"], df["gz"] = df.g.galactic.cartesian.xyz.value
    df["gvx"], df["gvy"], df["gvz"] = df.g.galactic.velocity.d_xyz.value


def reconstruct_df_from_stanfit(stanfit):
    """Reconstruct pandas DataFrame with Gaia column names from a StanFit object."""
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


def plot_violins(azfit, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        gs = fig.add_gridspec(9, 2)
        axx, axy, axz = list(map(fig.add_subplot, (gs[0, 0], gs[1, 0], gs[2, 0])))
        ax2 = fig.add_subplot(gs[3:6, 0])
        ax3 = fig.add_subplot(gs[6:9, 0])
        ax4 = fig.add_subplot(gs[:, 1])
    else:
        (axx, axy, axz, ax2, ax3, ax4) = fig.axes

    pars = ["v0", "sigv", "T_param"]

    v0_samples = get_flat2d(azfit, "v0")
    axx.violinplot(v0_samples[:, 0], vert=False, showextrema=False)
    axy.violinplot(v0_samples[:, 1], vert=False, showextrema=False)
    axz.violinplot(v0_samples[:, 2], vert=False, showextrema=False)
    # ax1.set_yticks([1, 2, 3])
    # ax1.set_yticklabels(["$v_{0,x}$", "$v_{0,y}$", "$v_{0,z}$"][::-1])
    # ax1.set_xlabel(r"$\rm km\,\rm s^{-1}$")
    axx.set_title("mean velocity")

    ax2.violinplot(
        get_flat2d(azfit, "sigv"), positions=np.r_[1:4:1], vert=False, showextrema=False
    )
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(["$\sigma_{v,x}$", "$\sigma_{v,y}$", "$\sigma_{v,z}$"])
    ax2.set_xlabel(r"$\rm km\,\rm s^{-1}$")
    ax2.set_title("scale of $\Sigma$")

    ax3.violinplot(
        get_flat2d(azfit, "Omega")[:, [1, 2, 5]],
        positions=np.r_[1:4:1],
        vert=False,
        showextrema=False,
    )
    ax3.set_yticks([1, 2, 3])
    ax3.set_yticklabels(["$\Omega_{xy}$", "$\Omega_{xz}$", "$\Omega_{yz}$"])
    ax3.set_title("correlation of $\Sigma$")
    ax3.set_xlim(-1, 1)

    ax4.violinplot(
        get_flat2d(azfit, "T_param"),
        positions=np.r_[1:10:1],
        vert=False,
        showextrema=False,
    )
    ax4.set_yticks(np.arange(1, 10))
    ax4.set_yticklabels(["11", "12", "13", "21", "22", "23", "31", "32", "33"])
    ax4.set_xlabel(r"$\rm m\,\rm s^{-1}\,\rm pc^{-1}$")
    ax4.set_title("velocity gradient $\mathbf {T}_{ij}$")
    ax4.set_xlim(-50, 50)
    ax4.axvline(0, c="k")

    # for cax in fig.axes:
    #     cax.axvline(0, c="k")
    return fig


def plot_mean_velocity(azfit, fig=None, label=None):
    if fig is None:
        fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
    else:
        ax = fig.axes

    v0_samples = get_flat2d(azfit, "v0")
    for i, cax in enumerate(ax):
        sns.kdeplot(v0_samples[:, i], ax=cax, legend=False, label=label)
    ax[0].set_xlabel("$v_x$ [$\mathrm{km}\,\mathrm{s}^{-1}$]")
    ax[1].set_xlabel("$v_y$ [$\mathrm{km}\,\mathrm{s}^{-1}$]")
    ax[2].set_xlabel("$v_z$ [$\mathrm{km}\,\mathrm{s}^{-1}$]")
    fig.suptitle("mean velocity $v_0$", size=20)
    if len(ax[1].get_legend_handles_labels()[0]) > 0:
        ax[1].legend()
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def confidence_ellipse(
    x=None, y=None, cov=None, ax=None, n_std=3.0, facecolor="none", **kwargs
):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    cov : array_like, shape (2, 2)
        covariance matrix. Mutually exclusive with input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x is None and y is None:
        if cov is None:
            raise ValueError("Either ")
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_flat2d(azfit, param):
    """
    Extract (nsample, ...) 2d array from azfit

    azfit : az.InferenceData
        azfit object
    param : str
        paramter name

    Returns (nsamples, ...) array. The rest of the shape is determined by
    the shape of param, e.g.,:
        v0 -> (nsamples, 3)
        T_param -> (nsamples, 9)
    """
    tmp = np.swapaxes(azfit.posterior[param].stack(i=["chain", "draw"]).values, -1, 0)
    tmp = tmp.reshape((tmp.shape[0], -1))
    return tmp


def compare_param(azfit_dict, var_names, **kwargs):
    """Compare parameter summary for multiple fits

    Returns MuliIndex DataFrame
    """
    dfs = [az.summary(cfit, var_names) for k, cfit in azfit_dict.items()]
    return pd.concat(dfs, keys=azfit_dict.keys())
