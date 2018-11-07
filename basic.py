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


model = kinesis.get_model('isotropic_pm')

# %% Generate mock data
b0 = [17.7, 41.2, 13.3] * u.pc
v0 = [-6.32, 45.24, 5.30]
sigv = 0.3
N = 500


cl = kinesis.Cluster(v0, sigv, ws=[0,0.05,-0.05,0,0], b0=b0.value)\
    .sample_sphere(N=N, Rmax=10)
print(cl)

# take random errors from actual data
d = pd.read_csv("/Users/semyeong/projects/spelunky/oh17-dr2/dr2_vL_clusters_full.csv")\
    .groupby('cluster').get_group('Hyades')
idx = np.random.randint(0, high=len(d), size=N)
# d = pd.read_csv("data/reino_tgas_full.csv")
# idx = np.random.randint(0, high=len(d), size=N)
# print(d[['parallax_error', 'pmra_error', 'pmdec_error']].describe())
gaia_error_columns = [
    'parallax_error', 'pmra_error', 'pmdec_error',
    'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']
df = df.assign(**{col: d.loc[idx, col].values for col in gaia_error_columns})



# %% noisify
C = np.zeros((N, 3, 3))
# parallax_error, pm_error = 1, 1
# p = 1e3/np.linalg.norm(b0)
# print("velocity error =", np.hypot(pm_error/p, parallax_error/p**2*p
#     vdec_error = np.hypot(df.pmdec_error/df.parallax,
#                           df.parallax_error/df.parallax**2*df.pmdec)*4.74

# median erros of Reino 2018 sample
# C[:, 0, 0] = 0.3**2
# C[:, 1, 1] = 0.15**2
# C[:, 2, 2] = 0.1**2

# # true        [-6.32, 45.24, 5.3] 0.3
# optimizing  [-7.3030224  45.75921844  5.35065223] 0.264934178042987
# opt - true  [-0.9830224   0.51921844  0.05065223]
# sampling mu [-7.25065737 45.84577698  5.38234859] 0.3150747104379578
# mu - true   [-0.93065737  0.60577698  0.08234859]
# sampling sd [0.15160619 0.34876795 0.11664491] 0.014878611447490215

#
# C[:, 0, 0] = 1.
# C[:, 1, 1] = 1.
# C[:, 2, 2] = 1.

# median DR2
# C[:, 0, 0] = 0.075**2
# C[:, 1, 1] = 0.15**2
# C[:, 2, 2] = 0.1**2

# assert np.array_equal(C[0], np.eye(3))
C[:, [0,1,2], [0,1,2]] = df[['parallax_error', 'pmra_error', 'pmdec_error']].values**2
C[:, [0, 1], [1, 0]] = (df['parallax_error']*df['pmra_error']*df['parallax_pmra_corr']).values[:, None]
C[:, [0, 2], [2, 0]] = (df['parallax_error']*df['pmdec_error']*df['parallax_pmdec_corr']).values[:, None]
C[:, [1, 2], [2, 1]] = (df['pmra_error']*df['pmdec_error']*df['pmra_pmdec_corr']).values[:, None]

a_data = np.zeros((N, 3))
for i in range(N):
    a_data[i] = np.random.multivariate_normal(
        df[['parallax', 'pmra', 'pmdec']].values[i], C[i])



# %% FIT
def init():
    return dict(
        d=1e3/df.parallax.values,
        sigv=0.5,
        v0=v0
    )



r = model.optimizing(
    data=dict( N=N, ra=df.ra.values, dec=df.dec.values, a=a_data, C=C ),
    init=init,
    tol_param=1e-12, verbose=True, history_size=1)



# deltad_sigma = (r['d'] - (1e3/df.parallax.values))/df.parallax_error.values
# plt.hist(deltad_sigma, 32);


# %% sample
if 1:
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
    print(rs['v0'].std(axis=0), rs['sigv'].std())
    from corner import corner

    corner(np.hstack([rs['v0'], rs['sigv'][:,None]]), truths=v0+[2.]);


print('true       ', v0, sigv)
print('optimizing ', r['v0'], r['sigv'])
print('opt - true ', r['v0']-np.array(v0))
print('sampling mu', rs['v0'].mean(axis=0), rs['sigv'].mean())
print('mu - true  ', rs['v0'].mean(axis=0)-np.array(v0))
print('sampling sd', rs['v0'].std(axis=0), rs['sigv'].std())
