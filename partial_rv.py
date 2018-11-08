""" kinematic models of clusters """
# %% import
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astropy.units as u
import astropy.coordinates as coord

# # NOTE: To prevent output from pystan, this must happen before import pystan
# import logging

# logger.setLevel(logging.ERROR)
#
# # add root logger (logger Level always Warning)
# # not needed if PyStan already imported
# logger.addHandler(logging.NullHandler())
#
# logger_path = "pystan.log"
# fh = logging.FileHandler(logger_path, encoding="utf-8")
# fh.setLevel(logging.INFO)
# # optional step
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# logger.addHandler(fh)

import kinesis

np.random.seed(1038479)

model = kinesis.get_model('general_model')
# model = kinesis.get_model('isotropic_pm')


# %% Generate mock data
b0 = [17.7, 41.2, 13.3] * u.pc
v0 = np.array([-6.32, 45.24, 5.30])
sigv = 0.3
N = 500


cl = kinesis.Cluster(v0, sigv, ws=[0,0,0,0,0], b0=b0.value)\
    .sample_sphere(N=N, Rmax=5)

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

df = cl.members.df
df = df.assign(**{col: d.loc[idx, col].values for col in gaia_error_columns})

df['radial_velocity_error'] = .01

print(df.radial_velocity.std(), df.radial_velocity.mean() )

if 1:
    # %% noisify
    C = np.zeros((N, 3, 3))

    # # assert np.array_equal(C[0], np.eye(3))
    # C[:, [0,1,2], [0,1,2]] = df[['parallax_error', 'pmra_error', 'pmdec_error']].values**2
    # C[:, [0, 1], [1, 0]] = (df['parallax_error']*df['pmra_error']*df['parallax_pmra_corr']).values[:, None]
    # C[:, [0, 2], [2, 0]] = (df['parallax_error']*df['pmdec_error']*df['parallax_pmdec_corr']).values[:, None]
    # C[:, [1, 2], [2, 1]] = (df['pmra_error']*df['pmdec_error']*df['pmra_pmdec_corr']).values[:, None]

    C[:, 0, 0] = 0.01
    C[:, 1, 1] = 0.01
    C[:, 2, 2] = 0.01

    C[:, 0, 0] = 1
    C[:, 1, 1] = 1
    C[:, 2, 2] = 1


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


    r0 = model.optimizing(
        data=dict(
            N=N, ra=df.ra.values, dec=df.dec.values, a=a_data, C=C,
            Nrv=0, rv=[], rv_error=[],
            irv=np.array([]).astype(int)   #NOTE: dtype matters!
            ),
        init=init,
        tol_param=1e-12, history_size=1)


    Nrv = int(N*0.5)
    irand = np.random.choice(np.arange(N), size=Nrv, replace=False)
    print(Nrv, irand[:3], np.max(irand))
    rp = model.optimizing(
        data=dict(
            N=N, ra=df.ra.values, dec=df.dec.values, a=a_data, C=C,
            Nrv=Nrv,
            rv=df.radial_velocity.values[irand],
            rv_error=df.radial_velocity_error.values[irand],
            irv=irand
            ),
        init=init,
        tol_param=1e-12, history_size=1)


    r = model.optimizing(
        data=dict(
            N=N, ra=df.ra.values, dec=df.dec.values, a=a_data, C=C,
            Nrv=N, rv=df.radial_velocity.values, rv_error=df.radial_velocity_error.values,
            irv=np.arange(N)
            ),
        init=init,
        tol_param=1e-12, history_size=1)

    print(v0, sigv)
    print('No RV        :', r0['v0'], r0['sigv'])
    print('No RV  delta :', r0['v0']-v0, r0['sigv']-sigv)
    print('No RV p      :', rp['v0'], rp['sigv'])
    print('No RV p delta:', rp['v0']-v0, rp['sigv']-sigv)
    print('all RV       :', r['v0'], r['sigv'])
    print('all RV delta :', r['v0']-v0, r['sigv']-sigv)
