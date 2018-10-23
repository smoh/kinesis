""" kinematic modelling of clusters """

#
import numpy as np
import theano.tensor as tt
import pymc3 as pm

# Define model

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1)
    obs = pm.Normal('obs', mu=mu, sd=1, observed=np.random.randn(100))

    dist = pm.
    v0 = pm.Normal('v0', mu=0, sd=30, shape=3)
    sigv = pm.Uniform('sigv', lower=0, upper=50)
