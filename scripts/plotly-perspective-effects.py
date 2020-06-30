"""
Script to make a plot demonstrating perspective effects using plotly.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import kinesis as kn
import gapipes as gp


def add_pqr(ra, dec, fig, color, text):
    """
    Add tangent space axes.

    ra, dec (float): degrees
    fig (go.Figure): plotly figure
    """
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    r = 1.0
    xyz_s = np.r_[
        [r * np.cos(dec) * np.cos(ra), r * np.cos(dec) * np.sin(ra), r * np.sin(dec)]
    ]
    phat = np.r_[[-np.sin(ra), np.cos(ra), 0.0]]
    qhat = np.r_[[-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]]
    rhat = np.r_[[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]]

    length = 0.3
    ra_axis_line = np.vstack([xyz_s, xyz_s + phat * length]).T
    dec_axis_line = np.vstack([xyz_s, xyz_s + qhat * length]).T
    r_axis_line = np.vstack([xyz_s, xyz_s + rhat * length]).T

    for l in (ra_axis_line, dec_axis_line, r_axis_line):
        fig.add_trace(
            go.Scatter3d(
                x=l[0],
                y=l[1],
                z=l[2],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[l[0][0]],
                y=[l[1][0]],
                z=[l[2][0]],
                text=[text],
                mode="text",
                textfont=dict(color=color, size=15),
                showlegend=False,
            )
        )
    return fig


def add_sphere(fig, trace_kw=dict()):
    u = np.linspace(-np.deg2rad(180.0), np.deg2rad(180.0), 20)
    v = np.linspace(0, np.deg2rad(180.0), 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(
        go.Surface(z=z, x=x, y=y, opacity=0.3, hoverinfo="skip", showscale=False),
        **trace_kw
    )


def add_zerolines(fig, trace_kw=dict()):
    axis_lim = 1.0
    fig.add_trace(
        go.Scatter3d(
            x=[-axis_lim, axis_lim],
            y=[0, 0],
            z=[0, 0],
            mode="lines",
            line=dict(color="black",),
            showlegend=False,
        ),
        **trace_kw
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[-axis_lim, axis_lim],
            z=[0, 0],
            mode="lines",
            line=dict(color="black",),
            showlegend=False,
        ),
        **trace_kw
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-axis_lim, axis_lim],
            mode="lines",
            line=dict(color="black",),
            showlegend=False,
        ),
        **trace_kw
    )


def add_3d_arrow(xyz, uvw, fig, color="black", scale=1.0, frac=0.7, trace_kw=dict()):
    """
    xyz (tuple): position
    uvw (tuple): velocity
    scale (float): velocity is multiplied by this vector
    frac (float): fraction of the length that will be arrow head (Cone).
        Between 0 and 1.
    """
    x, y, z = xyz
    u, v, w = uvw
    x, y, z = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
    u, v, w = np.atleast_1d(u), np.atleast_1d(v), np.atleast_1d(w)

    fig.add_trace(
        go.Scatter3d(
            x=np.concatenate((x, x + u * scale * frac)),
            y=np.concatenate((y, y + v * scale * frac)),
            z=np.concatenate((z, z + w * scale * frac)),
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
        ),
        **trace_kw
    )

    fig.add_trace(
        go.Cone(
            x=x + u * frac * scale,
            y=y + v * frac * scale,
            z=z + w * frac * scale,
            u=u * (1 - frac) * scale,
            v=v * (1 - frac) * scale,
            w=w * (1 - frac) * scale,
            anchor="tail",
            showscale=False,
            colorscale=[[0, color], [1, color]],  # HACK
            sizemode="absolute",
        ),
        **trace_kw
    )


def make_grid_cluster(ra, dec, distance, v0, degsize=5.0):
    b0 = coord.ICRS(ra * u.deg, dec * u.deg, d * u.pc).cartesian.xyz.value

    ra_bins = np.linspace(ra - degsize, ra + degsize, 11)
    dec_bins = np.linspace(dec - degsize, dec + degsize, 11)
    ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)
    ra_grid = ra_grid.ravel()
    dec_grid = dec_grid.ravel()
    memicrs = coord.ICRS(ra_grid * u.deg, dec_grid * u.deg, [d] * ra_grid.size * u.pc)

    cl = kn.Cluster(v0, 0.0, b0=b0).sample_at(memicrs)
    return cl


ra = np.array([45, 300, 340.0])
dec = np.array([45, 45.0, -65.0])
d = 100

b0tmp = coord.ICRS(
    ra[0] * u.deg,
    dec[0] * u.deg,
    d * u.pc,
    0.0 * u.mas / u.yr,
    0.0 * u.mas / u.yr,
    10 * u.km / u.s,
)
v0 = b0tmp.velocity.d_xyz.value
print(v0)


fig = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{"type": "surface"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
    subplot_titles=[""]
    + [
        r"$\text{{Position {}}}:\, (\alpha,\,\delta)=({:.0f},\,{:.0f})$".format(i, *t)
        for i, t in enumerate(zip(ra, dec), start=1)
    ],
)

add_sphere(fig, trace_kw=dict(row=1, col=1))
add_zerolines(fig, trace_kw=dict(row=1, col=1))

colors = ["#2269c4", "#cc6904", "#439064"]
for cra, cdec, rowcol, color, poslabel in zip(
    ra,
    dec,
    [dict(row=1, col=2), dict(row=2, col=1), dict(row=2, col=2)],
    colors,
    ["1", "2", "3"],
):
    add_pqr(cra, cdec, fig, color, poslabel)
    xyz = (
        np.cos(np.deg2rad(cdec)) * np.cos(np.deg2rad(cra)),
        np.cos(np.deg2rad(cdec)) * np.sin(np.deg2rad(cra)),
        np.sin(np.deg2rad(cdec)),
    )
    add_3d_arrow(xyz, v0, fig, scale=0.05)

    cl = make_grid_cluster(cra, cdec, d, v0)
    c = cl.members.truth.g

    b0 = coord.ICRS(cra * u.deg, cdec * u.deg, d * u.pc).cartesian.xyz.value
    cc = coord.ICRS(
        *(b0 * u.pc),
        *(v0 * u.km / u.s),
        representation_type=coord.CartesianRepresentation,
        differential_type=coord.CartesianDifferential
    )
    # NOTE:cos(dec) factor is not applied when differential is accessed this way.
    vra0 = (
        cc.spherical.differentials["s"].d_lon.value
        * d
        / 1e3
        * gp.accessors._tokms
        * np.cos(np.deg2rad(cdec))
    )
    vdec0 = cc.spherical.differentials["s"].d_lat.value * d / 1e3 * gp.accessors._tokms

    tmpfig = ff.create_quiver(
        c.icrs.ra.value,
        c.icrs.dec.value,
        c.vra.values - vra0,
        c.vdec.values - vdec0,
        scale=1.0,
        line=dict(color="black"),
        showlegend=False,
    )

    fig.add_trace(tmpfig.data[0], **rowcol)


scale = 1.0
frac = 0.9
# add_3d_arrow(
#     (0.3, 0.4, 0.5), (0.1, 0.2, 0.3), fig,
# )
# add_3d_arrow((0.0, 0.0, 0.0), (0.1, 0.2, 0.3), fig, color="red")
camera = dict(eye=dict(x=0.5 * 2, y=-0.5 * 2, z=0.5 * 2))

fig.update_layout(
    # autosize=False,
    width=1000,
    height=800,
    # margin=dict(l=65, r=50, b=65, t=90),
    hovermode=False,
    scene=dict(
        xaxis=dict(zeroline=False, showbackground=False, showgrid=False),
        yaxis=dict(zeroline=False, showbackground=False, showgrid=False),
        zaxis=dict(zeroline=False, showbackground=False, showgrid=False),
    ),
    scene_camera=camera,
)
fig.update_xaxes(
    showline=True,
    showgrid=False,
    linewidth=1,
    linecolor="black",
    mirror=True,
    title_text="R.A. [deg]",
    title_font=dict(size=20),
    ticks="inside",
    ticklen=10,
)
fig.update_yaxes(
    showline=True,
    showgrid=False,
    linewidth=1,
    linecolor="black",
    mirror=True,
    title_text="Decl. [deg]",
    title_font=dict(size=20),
    ticks="inside",
    ticklen=10,
)

fig.update_xaxes(range=[292, 308], row=2, col=1)

fig.write_image("fig1.pdf")
# fig.show()
