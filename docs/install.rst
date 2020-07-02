Installing kinesis
==================

There are three major dependencies for kinesis:

- `pystan`_
- `astropy`_
- `arviz`_

The core functionality of kinesis depends on
`pystan`_,
which provides python interface to `stan <https://mc-stan.org>`_.
Pystan requires C++ compiler: please consult
`Pystan documentation <https://pystan.readthedocs.io/en/latest/installation_beginner.html>`_
if you have any issues to get it up and running.
It is always good to check that your installation of pystan works as expected
with, for example, the "eight schools" example available
on `this page <https://pystan.readthedocs.io/en/latest/getting_started.html#example-1-eight-schools>`_.

Additionally, you need astropy>=3.0 with velocity support in astropy.coordinates.
I would generally recommend a sensible latest release of astropy as it is now on version 4.0.

Finally, `arviz`_ is useful for analyzing the posterior samples.

.. _pystan: https://pystan.readthedocs.io/en/latest/index.html 
.. _astropy: http://astropy.org
.. _arviz: https://arviz-devs.github.io/arviz/