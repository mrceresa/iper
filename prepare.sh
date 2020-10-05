
conda create -n iper -y python=3.8 fiona geopandas matplotlib ipython numpy scipy pandas lxml cython coloredlogs seaborn
conda activate iper

pip install contextily
pip install git+https://github.com/projectmesa/mesa#egg=mesa
pip install git+https://github.com/Corvince/mesa-geo.git#egg=mesa-geo

pip install -e .

pip install covid19dh lmfit==1.0.0

