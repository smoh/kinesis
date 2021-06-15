#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import kinesis as kn
import pystan

np.random.seed(73929)


# def test_models_compile():
#     for model_name in kn.models.available:
#         model = kn.get_model(model_name, recompile=True)
#         assert isinstance(model, pystan.StanModel)


def test_model():
    N, Nbg = 40, 10  # number of sources
    b0 = np.array([17.7, 41.2, 13.3])  # pc
    v0 = np.array([-6.32, 45.24, 5.30])  # [vx, vy, vz] in km/s
    sigv = 0.0  # dispersion, km/s
    cl = (
        kn.Cluster(v0, sigv, b0=b0, omegas=[100, 0, 200])
        .sample_sphere(N=N, Rmax=5)
        .observe(cov=np.eye(3) * 0.01, rv_error=np.ones(N))
    )
    bg = (
        kn.Cluster(v0, 10.0, b0=b0)
        .sample_sphere(N=Nbg, Rmax=5)
        .observe(cov=np.eye(3) * 0.01, rv_error=np.ones(Nbg))
    )
    df = pd.concat((cl.members.observed, bg.members.observed), axis=0)
    m = kn.AllCombined(recompile=True)

    def initfunc():
        return dict(
            v0=np.array(v0) + np.random.normal(size=3),
            sigv=[2, 2, 2],
            sigv_bg=50,
            f_mem=0.6,
        )

    fit = m.fit(df, b0=b0, init=initfunc)
