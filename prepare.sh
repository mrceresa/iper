#!\bin\bash
set -xe
## Create iper environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda create -c conda-forge -n iper -y python=3.10 mamba

conda activate iper

mamba install -c conda-forge -y matplotlib numpy scipy pandas seaborn coloredlogs lxml cython   
mamba install -c conda-forge -y fiona geopandas geoplot pyproj rtree shapely descartes
mamba install -c conda-forge -y contextily trimesh meshio transitions 

#pip install contextily trimesh meshio transitions

pip install -U mesa
pip install -U mesa-geo

## Install iper in devel mode
pip install -e .

## Install packages needed for examples:
# SARS-COVID
pip install covid19dh lmfit==1.0.0

