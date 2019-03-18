""" kinematic models of clusters """
# %% import
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import os, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan

import astropy.units as u
import astropy.coordinates as coord

import kinesis

np.random.seed(1038479)

model = kinesis.get_model('isotropic_pm')

# %% Generate mock data
b0 = [17.7, 41.2, 13.3] * u.pc
v0 = [-6.32, 45.24, 5.30]
sigv = 0.3
N = 500


cl = kinesis.Cluster(v0, sigv, b0=b0.value)\
    .sample_sphere(N=N, Rmax=1)
print(cl)
df = cl.members.df

# %% noisify
C = np.zeros((N, 3, 3))
C[:, 0, 0] = 0.3**2
C[:, 1, 1] = 0.15**2
C[:, 2, 2] = 0.1**2

a_data = np.zeros((N, 3))
for i in range(N):
    a_data[i] = np.random.multivariate_normal(
        df[['parallax', 'pmra', 'pmdec']].values[i], C[i])


def init():
    return dict(
        d=1e3/df.parallax.values,
        sigv=0.5,
        v0=v0
    )

r = model.optimizing(
    data=dict(N=N, ra=df.ra.values, dec=df.dec.values, a=a_data, C=C),
    init=init,
    tol_param=1e-12, verbose=True, history_size=1)
print('true       ', v0, sigv)
print('optimizing ', r['v0'], r['sigv'])
print('opt - true ', r['v0']-np.array(v0))





# # %% sample
if 0:
    def new_init():
        return dict(
            d=1./df.parallax.values * 1000.,
            sigv=r['sigv'],
            v0=r['v0']
        )
    rs = model.sampling(data=dict(
        N=N,
        ra=df.ra.values,
        dec=df.dec.values,
        a=df[['parallax', 'pmra', 'pmdec']].values,
        C=C
    ), init=new_init)

    print('sampling')
    print(rs['v0'].mean(axis=0), rs['sigv'].mean())
#     print(rs['v0'].std(axis=0), rs['sigv'].std())
#     from corner import corner
#
#     corner(np.hstack([rs['v0'], rs['sigv'][:,None]]), truths=v0+[2.]);
#

    print('true       ', v0, sigv)
    print('optimizing ', r['v0'], r['sigv'])
    print('opt - true ', r['v0']-np.array(v0))
    print('sampling mu', rs['v0'].mean(axis=0), rs['sigv'].mean())
    print('mu - true  ', rs['v0'].mean(axis=0)-np.array(v0))
    print('sampling sd', rs['v0'].std(axis=0), rs['sigv'].std())
