''' Play with the hyades cluster in many projected space
'''

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.coordinates as coord

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider

import kinesis

rootdir = '/Users/semyeong/projects/kinesis/'

hy_tgas = pd.read_csv(rootdir+"reino_tgas_full.csv")
# full DR2 table for Hyades stars from DR2 van Leeuwen 2018
hy_dr2 = pd.read_csv("/Users/semyeong/projects/spelunky/oh17-dr2/dr2_vL_clusters_full.csv")\
    .groupby('cluster').get_group('Hyades')
hy_tgas = hy_tgas.pipe(kinesis.add_vtan).pipe(kinesis.add_vtan_errors)
hy_dr2 = hy_dr2.pipe(kinesis.add_vtan).pipe(kinesis.add_vtan_errors)


mockhy = kinesis.Cluster([-5.96, 45.60, 5.57], 0.3, omegas=[0,0,0.],
                         ws=[.1,0,0,0,0],
                         b0=[17.2, 41.6, 13.6])
mockhy_members = mockhy.sample_at(kinesis.make_icrs(hy_dr2, coord.ICRS))

source = ColumnDataSource(data=dict(
    ra=hy_dr2.ra,
    dec=hy_dr2.dec,
    parallax=hy_dr2.parallax,
    pmra=mockhy_members.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value,
    pmdec=mockhy_members.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value))

vmean = [-5.96, 45.60, 5.57]
vx = Slider(title='vx', value=vmean[0], start=-50, end=50)
vy = Slider(title='vy', value=vmean[1], start=-50, end=50)
vz = Slider(title='vz', value=vmean[2], start=-50, end=50)
sigmav = Slider(title='sigmav', value=0, start=0, end=2, step=0.1)
omegax = Slider(title='omega_x', value=0, start=-1, end=1, step=0.1)
omegay = Slider(title='omega_y', value=0, start=-1, end=1, step=0.1)
omegaz = Slider(title='omega_z', value=0, start=-1, end=1, step=0.1)

def update_data(attrname, old, new):
    cvx = vx.value
    cvy = vy.value
    cvz = vz.value
    csigmav = sigmav.value
    comegax = omegax.value
    comegay = omegay.value
    comegaz = omegaz.value

    mockhy = kinesis.Cluster([cvx, cvy, cvz], csigmav, omegas=[comegax, comegay, comegaz],
                             ws=[0, 0, 0, 0, 0],
                             b0=[17.2, 41.6, 13.6])
    mockhy_members = mockhy.sample_at(kinesis.make_icrs(hy_dr2, coord.ICRS))

    source.data = dict(
        ra=hy_dr2.ra,
        dec=hy_dr2.dec,
        parallax=hy_dr2.parallax,
        pmra=mockhy_members.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value,
        pmdec=mockhy_members.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value)

for w in [vx, vy, vz, sigmav, omegax, omegay, omegaz]:
    w.on_change('value', update_data)


p = figure(plot_height=400, plot_width=400)
p.xaxis.axis_label = 'R.A.'
p.yaxis.axis_label = 'Decl.'
p.circle('ra', 'dec', source=source)

p_pm = figure(plot_height=400, plot_width=400)
p_pm.xaxis.axis_label = 'pmra'
p_pm.yaxis.axis_label = 'pmdec'
p_pm.circle('pmra', 'pmdec', source=source)

inputs = widgetbox(vx, vy, vz, sigmav, omegax, omegay, omegaz)
curdoc().add_root(row(inputs, p, p_pm))

