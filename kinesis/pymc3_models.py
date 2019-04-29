""" models implemented in pymc3 """
import pymc3 as pm
import theano
import theano.tensor as tt


import os
import pickle
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
N = 50


cl = kinesis.Cluster(v0, sigv, b0=b0.value)\
    .sample_sphere(N=N, Rmax=1)
print(cl)
df = cl.members.df

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
C[:, 0, 0] = 0.3**2
C[:, 1, 1] = 0.15**2
C[:, 2, 2] = 0.1**2

a_data = np.zeros((N, 3))
for i in range(N):
    a_data[i] = np.random.multivariate_normal(
        df[['parallax', 'pmra', 'pmdec']].values[i], C[i])




M = theano.shared(np.ones((N, 2, 3)))
cov = theano.shared(np.repeat(np.eye(3)[None,:], N, axis=0))
a_data = theano.shared(np.zeros([N, 3]))

with pm.Model() as model:
    d = pm.HalfNormal('d', sd=10, shape=N)
    v0 = pm.MvNormal('v0', mu=np.zeros(3), cov=np.eye(3), shape=3, testval=np.zeros(3))
    sigv = pm.HalfNormal('sigv', sd=5)

    # M has shape (N, 2, 3)
    a_model = pm.Deterministic(
        'a_model',
        tt.concatenate([
            1e3/d[:,np.newaxis],
            M.dot(v0.T) / (d[:,np.newaxis] / 1e3) / 4.74], axis=1))

    # TODO: there must be a better way?
    a_obs = [pm.MvNormal('a_obs_{:d}'.format(i), mu=a_model[i], cov=cov[i],
                         observed=a_data[i]) for i in range(N)]
