"""
Plot a diagram showing receding clusters contract and approaching clusters expand.
"""
import matplotlib.pyplot as plt
import numpy as np
import kinesis as kn

kn.set_mpl_style()

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(7, 2.5),
    # subplot_kw=dict(aspect="equal"),
    # gridspec_kw=dict(width_ratios=[1.33, 1.0]),
)


v = [0, 0.3]

for v, cax in zip([[0, 0.2], [0, -0.2]], (ax1, ax2)):

    thetap = np.linspace(-np.deg2rad(30), np.deg2rad(30), 101)
    theta = np.pi / 2 - thetap
    cax.plot(np.cos(theta), np.sin(theta), lw=4, alpha=0.5)
    # at sparse grid, plot projected component
    thetap = np.linspace(-np.deg2rad(20), np.deg2rad(20), 9)
    theta = np.pi / 2 - thetap
    theta_hat = np.array([[-np.sin(t), np.cos(t)] for t in theta])
    vproj = np.einsum("ni,i->n", theta_hat, v)[:, None] * theta_hat
    x = np.cos(theta)
    y = np.sin(theta)
    cax.quiver(
        x, y, *v, scale=1, color="gray",
    )
    cax.quiver(x, y, *(vproj.T), zorder=10, scale=1)
    if v[1] > 0:
        cax.axis([-0.4, 0.4, 0.65, 1.3])
    else:
        cax.axis([-0.4, 0.4, 0.65, 1.3])

    cax.spines["right"].set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.spines["left"].set_visible(False)
    cax.spines["bottom"].set_visible(False)
    cax.set_xticks([])
    cax.set_yticks([])

fig.tight_layout()
fig.savefig("../report/perspective-radial.pdf")
