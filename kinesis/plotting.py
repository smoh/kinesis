import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

__all__ = [
    "mystyledict",
    "set_mpl_style",
    "confidence_ellipse",
    "plot_violins",
    "plot_mean_velocity",
]


mystyledict = {
    "axes.linewidth": 1.0,
    "axes.titlesize": 18.0,
    "axes.labelsize": 14,
    "figure.dpi": 120.0,
    "figure.figsize": [5, 4],
    "font.family": "serif",
    "font.serif": "Liberation Serif",
    #     'font.serif':'CMU Serif',
    'axes.unicode_minus':False,
    "font.size": 12.0,
    "grid.color": "tab:gray",
    "grid.linewidth": 0.7,
    "legend.fancybox": False,
    "legend.edgecolor": "0.1",
    "lines.linewidth": 1.0,
    "axes.formatter.use_mathtext": True,
    "xtick.bottom": True,
    "xtick.direction": "in",
    "xtick.major.pad": 7.0,
    "xtick.major.size": 8.0,
    "xtick.major.width": 1.0,
    "xtick.minor.size": 0.0,
    "xtick.minor.width": 1.5,
    "xtick.top": False,
    "ytick.direction": "in",
    "ytick.left": True,
    "ytick.major.pad": 7.0,
    "ytick.major.size": 8.0,
    "ytick.major.width": 1.0,
    "ytick.minor.size": 0.0,
    "ytick.minor.width": 1.5,
    "ytick.right": False,
    "mathtext.fontset": "cm",
    "text.usetex": True,
    "axes.unicode_minus": True,
}


def set_mpl_style():
    plt.style.use(mystyledict)


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
