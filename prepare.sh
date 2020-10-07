
## Create iper environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda create -n iper -y python=3.8 fiona geopandas matplotlib ipython numpy scipy pandas lxml cython coloredlogs seaborn
conda activate iper

pip install contextily
pip install git+https://github.com/projectmesa/mesa#egg=mesa
pip install git+https://github.com/Corvince/mesa-geo.git#egg=mesa-geo

## Install iper in devel mode
pip install -e .

## Install packages needed for examples:
# SARS-COVID
pip install covid19dh lmfit==1.0.0

