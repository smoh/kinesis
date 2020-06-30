
Models were developed in an inscreasing order of complexity.

- isotropic_pm.stan: proper motion only, v0=[vx, vy, vz], isotropic dispersion sigv
- general_model.stan: proper motion + paritial RVs, v0, isotropic dispersion sigv, and optionally T_ij = dv_i/dx_j
- anisotropic_rv.stan, anistropic_rv2.stan: proper motion + partial RVs, v0, anistropic dispersion parameterized in quad from (scale and correlation matrix), no velocity gradient.
- mixture.stan: extends general_model to include contamination as a mixture model


- allcombined.stan: combines mixture model, (optional) velocity gradient and anisotropic dispersion. This is the final model
- allcombined-fixed-v0.stan: only fixes mean velocity v0 to the given value (parameters -> data block)

- allcombined-rvoffset.stan: adds one additional parameter rvoffset which offsets RVs by this amount.
    The hope was that the systematic bias that affects RVs (without regards to the geometry due to e.g., spectroscopic analysis)
    may be constrained simultaneously but this was not tested.