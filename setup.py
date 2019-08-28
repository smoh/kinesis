#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup, Command

NAME = "kinesis"
DESCRIPTION = "Kinematic modelling of comoving stars"
AUTHOR = "Semyeong Oh"
EMAIL = "semyeong.oh@gmail.com"

setup(
    name=NAME,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(exclude=("tests",)),
    package_data={'kinesis': ['stan/*.stan']},
    install_requires=["astropy", "arviz", "pystan"],
)
