import os
from setuptools import find_packages, setup

# Important Paths
PROJECT = os.path.abspath(os.path.dirname(__file__))

# Directories to ignore in find_packages
EXCLUDES = ()

setup(
    name="nnflow",
    packages=find_packages(
        where=PROJECT, include=["nnflow", "nnflow.*"], exclude=EXCLUDES
    ),
)