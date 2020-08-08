# kinesis

Kinematic modelling of clusters with the [Gaia](https://www.cosmos.esa.int/web/gaia/home) data.

<a href="https://smoh.space/kinesis"><img src="https://github.com/smoh/kinesis/workflows/docs/badge.svg"></a>
[![codecov](https://codecov.io/gh/smoh/kinesis/branch/master/graph/badge.svg)](https://codecov.io/gh/smoh/kinesis)

Kinesis is a package for fitting the internal kinematics of a star cluster
with astrometry and (incomplete) radial velocity data of its members.
In the most general model, the stars can be a mixture of background (contamination)
and the cluster, for which the (3,3) velocity dispersion matrix and
velocity gradient (dv_x/dx, dv_y/dx, ...) are included.
Please refer to [Oh & Evans 2020](https://arxiv.org/abs/2007.02969) for full details.
There are also simpler versions of the most general model and
utilities to generate mock clusters and mock observations.

Check out the [documentation](https://smoh.space/kinesis) (it is under
development and may be incomplete).


## Attribution

If you make use of this code, please cite [Oh & Evans 2020](https://arxiv.org/abs/2007.02969).
The bibtex entry is:
```
@ARTICLE{2020arXiv200702969O,
       author = {{Oh}, Semyeong and {Evans}, N. Wyn},
        title = "{Kinematic modelling of clusters with Gaia: the Death Throes of the Hyades}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2020,
        month = jul,
          eid = {arXiv:2007.02969},
        pages = {arXiv:2007.02969},
archivePrefix = {arXiv},
       eprint = {2007.02969},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200702969O},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```