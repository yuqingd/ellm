conda create -n habitat6 python=3.6 cmake=3.14.0
conda activate habitat6

cd ~/Documents/hab2
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
conda config --add channels conda-forge
conda install -y bullet
python setup.py build_ext --parallel 2 install --with-cuda --bullet --headless

cd ~/Documents/cos-hab2
pip install -r pip_requirements.txt

cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all
cd -

cd rl-toolkit && pip install -e . && cd -
pip install -r requirements.txt

