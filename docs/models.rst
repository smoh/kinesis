Available models
================

This page lists different variants of the full model presented in the
`paper <#>`_, starting the most simplest.
Each model is written in one ``<model name>.stan`` file.
Use ``get_model`` function to get the model you want, which will
return pystan `StanModel <https://pystan.readthedocs.io/en/latest/api.html#pystan.StanModel>`_ instance.


.. code-block:: python

    import kinesis as kn
    model = kn.get_model('allcombined')


The first time you try to get each model, it will take a little bit as
`pystan <https://pystan.readthedocs.io/en/latest/index.html>`_ needs to
compile the translated C++ code. For future times, it will load instantly as
kinesis uses the pickled model.

Models were developed in an increasing order of complexity.

- isotropic_pm.stan: proper motion only, v0=[vx, vy, vz], isotropic dispersion sigv
- general_model.stan: proper motion + paritial RVs, v0, isotropic dispersion sigv, and optionally T_ij = dv_i/dx_j
- anisotropic_rv.stan, anistropic_rv2.stan: proper motion + partial RVs, v0, anistropic dispersion parameterized in quad from (scale and correlation matrix), no velocity gradient.
- mixture.stan: extends general_model to include contamination as a mixture model


- allcombined.stan: combines mixture model, (optional) velocity gradient and anisotropic dispersion. This is the final model
- allcombined-fixed-v0.stan: only fixes mean velocity v0 to the given value (parameters -> data block)
