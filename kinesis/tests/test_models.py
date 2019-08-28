#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import astropy.coordinates as coord
import astropy.units as u

import kinesis as kn
import pystan
import gapipes as gp

np.random.seed(73929)


def test_compile_general_model():
    model = kn.get_model("general_model")
    assert isinstance(model, pystan.StanModel)
