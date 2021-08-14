#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup, Command
import codecs
import os.path


# Ref: https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


NAME = "kinesis"
DESCRIPTION = "Kinematic modelling of comoving stars"
AUTHOR = "Semyeong Oh"
EMAIL = "semyeong.oh@gmail.com"


setup(
    name=NAME,
    version=get_version("kinesis/__init__.py"),
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(exclude=("tests",)),
    package_data={"kinesis": ["stan/*.stan"]},
    install_requires=["astropy", "arviz", "pystan==2"],
)
