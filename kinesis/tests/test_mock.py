#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import astropy.coordinates as coord
import astropy.units as u

from kinesis import mock


def test_sample_at():
    cl = mock.Cluster([12.3, 23.1, -3.5], 0.5)
    pos = coord.ICRS([21.3]*u.deg, [54]*u.deg, [95.2]*u.pc)
    cl.sample_at(pos)
