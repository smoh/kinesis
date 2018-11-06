#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from kinesis import pipes


def test_pipe():
    '''Test pipe decorator raises error'''
    @pipes.pipe
    def f(*args, **kwargs):
        pass
    with pytest.raises(ValueError):
        f(1)
