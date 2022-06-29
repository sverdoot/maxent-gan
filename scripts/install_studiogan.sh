#!/usr/bin/env bash

git clone https://github.com/POSTECH-CVLab/PyTorch-StudioGAN.git thirdparty/studiogan 
cd thirdparty/studiogan 
git checkout cce0c6ab9584deb8dbf289e6192c125b201aa3d6
mv src studiogan

touch studiogan/__init__.py

echo "import sys

sys.path.append('.')
from . import config, utils" >> studiogan/__init__.py

echo \
"from setuptools import setup, find_packages

setup(name='studiogan',
      version='1.0',
      packages=find_packages())" \
>> setup.py

pip install -e .

echo "tqdm ninja h5py kornia matplotlib pandas sklearn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg prdc" >> requirements.txt
cd ../..

poetry add `cat thirdparty/studiogan/requirements.txt`

ln -s thirdparty/studiogan/studiogan studiogan

