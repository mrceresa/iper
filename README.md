![Unit tests](https://github.com/bcn-medtech/iper-social-simulations/workflows/Unit%20tests/badge.svg)

# iper - a Social policy explorer for Agent-Based Modeling

This is the first release of iper. No functionality guaranteed, bugs included.

## Installation

The code works on Python 3.7+ To install iper on linux or macOS run

```shell
pip install -e git+https://github.com/bcn-medtech/iper-social-simulations.git#egg=iper
```

If you have some errors building C-extensions in the previous step, consider first using Anaconda to install some of the requirements with

```shell
conda install fiona pyproj rtree shapely mesa mesa-geo
```

## Getting started

After installing the package, clone the repository with
```shell
git clone https://github.com/bcn-medtech/iper-social-simulations.git
```

and run the example:
```shell
cd examples/SAR-COV2/
python run.py
```


