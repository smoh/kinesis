''' Play with the hyades cluster in many projected space
'''

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.coordinates as coord

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import row, widgetbox, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider

import kinesis

rootdir = '/Users/semyeong/projects/kinesis/data/'

hy_tgas = pd.read_csv(rootdir+"reino_tgas_full.csv")
# full DR2 table for Hyades stars from DR2 van Leeuwen 2018
hy_dr2 = pd.read_csv("/Users/semyeong/projects/spelunky/oh17-dr2/dr2_vL_clusters_full.csv")\
    .groupby('cluster').get_group('Hyades')
hy_tgas = hy_tgas.pipe(kinesis.add_vtan).pipe(kinesis.add_vtan_errors)
hy_dr2 = hy_dr2.pipe(kinesis.add_vtan).pipe(kinesis.add_vtan_errors)


mockhy = kinesis.Cluster([0,0,0], 0.3, omegas=[0,0,0.],
                         ws=[0,0,0,0,0],
                         b0=[17.2, 41.6, 13.6])
mockhy_members = mockhy.sample_at(kinesis.make_icrs(hy_dr2, coord.ICRS))

source = ColumnDataSource(data=dict(
    ra=hy_dr2.ra,
    dec=hy_dr2.dec,
    parallax=hy_dr2.parallax,
    pmra=mockhy_members.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value,
    pmdec=mockhy_members.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value,
    vra=mockhy_members.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value/hy_dr2.parallax*4.74,
    vdec=mockhy_members.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value/hy_dr2.parallax*4.74,

    x=mockhy_members.cartesian.x.to(u.pc).value,
    y=mockhy_members.cartesian.y.to(u.pc).value,
    z=mockhy_members.cartesian.z.to(u.pc).value
    ))

vx = Slider(title='vx', value=0, start=-50, end=50)
vy = Slider(title='vy', value=0, start=-50, end=50)
vz = Slider(title='vz', value=0, start=-50, end=50)
sigmav = Slider(title='sigmav', value=0.3, start=0, end=2, step=0.1)
omegax = Slider(title='omega_x', value=0, start=-1, end=1, step=0.1)
omegay = Slider(title='omega_y', value=0, start=-1, end=1, step=0.1)
omegaz = Slider(title='omega_z', value=0, start=-1, end=1, step=0.1)
w1 = Slider(title='w1', value=0, start=-0.5, end=0.5, step=0.1)
w2 = Slider(title='w2', value=0, start=-0.5, end=0.5, step=0.1)
w3 = Slider(title='w3', value=0, start=-0.5, end=0.5, step=0.1)
w4 = Slider(title='w4', value=0, start=-0.5, end=0.5, step=0.1)
w5 = Slider(title='w5', value=0, start=-0.5, end=0.5, step=0.1)

def update_data(attrname, old, new):
    cvx = vx.value
    cvy = vy.value
    cvz = vz.value
    csigmav = sigmav.value
    comegax = omegax.value
    comegay = omegay.value
    comegaz = omegaz.value
    cw1, cw2, cw3, cw4, cw5 = w1.value, w2.value, w3.value, w4.value, w5.value

    mockhy = kinesis.Cluster([cvx, cvy, cvz], csigmav, omegas=[comegax, comegay, comegaz],
                             ws=[cw1, cw2, cw3, cw4, cw5],
                             b0=[17.2, 41.6, 13.6])
    mockhy_members = mockhy.sample_at(kinesis.make_icrs(hy_dr2, coord.ICRS))

    source.data = dict(
        ra=hy_dr2.ra,
        dec=hy_dr2.dec,
        parallax=hy_dr2.parallax,
        pmra=mockhy_members.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value,
        pmdec=mockhy_members.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value,
        vra=mockhy_members.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value/hy_dr2.parallax*4.74,
        vdec=mockhy_members.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value/hy_dr2.parallax*4.74,
        x=mockhy_members.cartesian.x.to(u.pc).value,
        y=mockhy_members.cartesian.y.to(u.pc).value,
        z=mockhy_members.cartesian.z.to(u.pc).value)

for w in [vx, vy, vz, sigmav, omegax, omegay, omegaz, w1, w2, w3, w4, w5]:
    w.on_change('value', update_data)

hy_dr2_res = hy_dr2.copy().pipe(kinesis.add_xv, coord.ICRS)
vmean1 = [-5.96, 45.60, 5.57]
mean_cl = kinesis.Cluster(vmean1, 0)
mean_cl_members = mean_cl.sample_at(kinesis.make_icrs(hy_dr2, coord.ICRS))
hy_dr2_res.pmra -= mean_cl_members.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value
hy_dr2_res.pmdec -= mean_cl_members.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value
hy_dr2_res['dvra'] = hy_dr2_res.pmra/hy_dr2_res.parallax*4.74
hy_dr2_res['dvdec'] = hy_dr2_res.pmdec/hy_dr2_res.parallax*4.74


p = figure(plot_height=300, plot_width=300)
p.xaxis.axis_label = 'R.A.'
p.yaxis.axis_label = 'Decl.'
p.circle('ra', 'dec', source=source)
p.circle('ra', 'dec', source=hy_dr2_res, color='red')

p_pm = figure(plot_height=300, plot_width=300)
p_pm.xaxis.axis_label = 'pmra'
p_pm.yaxis.axis_label = 'pmdec'
p_pm.circle('pmra', 'pmdec', source=source)
p_pm.circle('pmra', 'pmdec', source=hy_dr2_res, color='red')

p3 = figure(plot_height=300, plot_width=300)
p3.xaxis.axis_label = 'dec'
p3.yaxis.axis_label = 'pmra'
p3.circle('dec', 'pmra', source=source)
p3.circle('dec', 'pmra', source=hy_dr2_res, color='red')

p4 = figure(plot_height=300, plot_width=300)
p4.xaxis.axis_label = 'ra'
p4.yaxis.axis_label = 'pmdec'
p4.circle('ra', 'pmdec', source=source)
p4.circle('ra', 'pmdec', source=hy_dr2_res, color='red')

p_zvra = figure(plot_height=300, plot_width=300)
p_zvra.xaxis.axis_label = 'z'
p_zvra.yaxis.axis_label = 'vra'
p_zvra.circle('z', 'vra', source=source)
p_zvra.circle('z', 'dvra', source=hy_dr2_res, color='red')

p_zvdec = figure(plot_height=300, plot_width=300)
p_zvdec.xaxis.axis_label = 'z'
p_zvdec.yaxis.axis_label = 'vdec'
p_zvdec.circle('z', 'vdec', source=source)
p_zvdec.circle('z', 'dvdec', source=hy_dr2_res, color='red')

p_xvra = figure(plot_height=300, plot_width=300)
p_xvra.xaxis.axis_label = 'x'
p_xvra.yaxis.axis_label = 'vra'
p_xvra.circle('x', 'vra', source=source)
p_xvra.circle('x', 'dvra', source=hy_dr2_res, color='red')

p_xvdec = figure(plot_height=300, plot_width=300)
p_xvdec.xaxis.axis_label = 'x'
p_xvdec.yaxis.axis_label = 'vdec'
p_xvdec.circle('x', 'vdec', source=source)
p_xvdec.circle('x', 'dvdec', source=hy_dr2_res, color='red')

inputs = widgetbox(vx, vy, vz, sigmav, omegax, omegay, omegaz, w1, w2, w3, w4, w5)
curdoc().add_root(row(inputs, gridplot(
    [[p, p_pm],
     [p_zvra, p_zvdec],
     [p_xvra, p_xvdec]
     ])))
