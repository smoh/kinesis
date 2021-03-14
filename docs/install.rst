Installing kinesis
==================

There are three major dependencies for kinesis:

- `pystan2`_
- `astropy`_
- `arviz`_

The core functionality of kinesis depends on
`pystan2`_,
which provides python interface to `stan <https://mc-stan.org>`_.
Pystan requires C++ compiler: please consult
`Pystan2 documentation <https://pystan2.readthedocs.io/en/latest/installation_beginner.html>`_
if you have any issues to get it up and running.
It is always good to check that your installation of pystan works as expected
with, for example, the "eight schools" example available
on `this page <https://pystan2.readthedocs.io/en/latest/getting_started.html#example-1-eight-schools>`_.

.. note::
    Pystan recently moved on to develop pystan3 but kinesis still requires pystan2 interface.

Additionally, you need astropy>=3.0 with velocity support in astropy.coordinates.
I would generally recommend a sensible latest release of astropy as it is now on version 4.0.

Finally, we will use `arviz`_ for analyzing the fits.

.. _pystan2: https://pystan2.readthedocs.io/
.. _astropy: http://astropy.org
.. _arviz: https://arviz-devs.github.io/arviz/
