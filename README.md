![Unit tests](https://github.com/bcn-medtech/iper-social-simulations/workflows/Unit%20tests/badge.svg)

# iper - a Social policy explorer for Agent-Based Modeling

This is the first release of iper. No functionality guaranteed, bugs included.

## Installation

The code works on Python 3.7+ To install iper on linux or macOS:

(1) First, install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). 

(2) Install git lfs to automatically download the shapefile data.

(3) Then clone the repository

(4) Launch prepare script that will configure a conda environment

```shell
git lfs install
git clone https://github.com/bcn-medtech/iper-social-simulations.git
cd iper-social-simulations/
./prepare.sh
```

## Getting started

After installing the package, run the examples:
```shell
cd examples/SAR-COV2/
python run.py
```

```shell
cd examples/SAR-COV2/0dim/
python seihrd.py fit
```


