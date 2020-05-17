#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from setuptools import setup, find_packages
from codecs import open

requires = ["mesa >= 0.8.6", "mesa-geo", "geopandas"]

version = ""
with open("iper/__init__.py", "r") as fd:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE
    ).group(1)

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="iper",
    version=version,
    description="Social simulator and policy explorer",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Mario Ceresa, BCNMedTech",
    author_email="",
    url="https://github.com/bcn-medtech/iper-social-simulations",
    packages=find_packages(),
    package_data={
    },
    include_package_data=False,
    install_requires=requires,
    keywords="agent based modeling model ABM social simulation multi-agent policy explorer reinforcement learning ",
    license="Apache 2.0",
    zip_safe=False,
    classifiers=(
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
    ),
)
