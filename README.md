# Experiments on sampling from GAN with constraints

- [Experiments on sampling from GAN with constraints](#experiments-on-sampling-from-gan-with-constraints)
  - [Getting started](#getting-started)
    - [StudioGAN:](#studiogan)
  - [TODO:](#todo)


link to gdrive with checkpoints and stats: https://drive.google.com/drive/folders/1bgIBVt5JAqb8RsHHDFRVp003YYjBoSXV?usp=sharing

## Getting started

Create environment and set dependencies:
```zsh
conda create -n constrained_gan python=3.8
```

```zsh
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.create false

conda activate constrained_gan
conda install tensorflow-gpu==2.4.1
poetry install
```

<!-- To compute FID in TF fashion:

```zsh
conda install tensorflow-gpu
``` -->

Check whether TF can see GPU:

```python
import tensorflow.compat.v1 as tf

tf.test.gpu_device_name()
```

If FID computation stucks that might be caused by conflict with scipy. Try running ```export MKL_NUM_THREADS=1```.

### StudioGAN:


```zsh
./scripts/install_studiogan.sh
```

<!-- ```zsh
git clone https://github.com/POSTECH-CVLab/PyTorch-StudioGAN.git thirdparty/studiogan 
cd thirdparty/studiogan 
git checkout cce0c6ab9584deb8dbf289e6192c125b201aa3d6
mv src studiogan
```

```zsh
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
```

```zsh
pip install -e .
```

```zsh
echo "tqdm ninja h5py kornia matplotlib pandas sklearn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg prdc" >> requirements.txt
cd ../..
```


```zsh 
poetry add `cat thirdparty/studiogan/requirements.txt`
```

Create symbolik link
```zsh
ln -s thirdparty/studiogan/studiogan studiogan
``` -->

To use StudioGAN as is following is needed:

```python
import sys
from pathlib import Path

from maxent_gan.utils.general_utils import ROOT_DIR

sys.path.append(Path(ROOT_DIR, 'studiogan'))
```
------------

```zsh
poetry install
```

<!-- To compute FID in TF fashion:

```zsh
wget  "https://raw.githubusercontent.com/bioinf-jku/TTUR/master/fid.py"  -P thirdparty/TTUR
``` -->


Put CIFAR-10 into directory ```data/cifar10```  using this script

```python
import torchvision.datasets as dset

cifar = dset.CIFAR10(root='data/cifar10', download=True)
```

Make bash scripts runable 

```zsh
chmod +x scripts/*.sh
```


<!-- ###
mann

```bash
git clone git@github.com:wzell/mann.git thirdparty
```

```bash
cd thirdparty/mann
pip install -r requirements.txt
echo \
"from setuptools import setup

setup(name='mann',
      version='1.0',
      packages=['mann'],
      package_dir={'mann': './models'},)
" \
>> setup.py

python setup.py install
``` -->



## Usage 

Download checkpoints:

<!-- ```bash
./scripts/get_ckpts.sh
./scripts/get_stats.sh
``` -->

```bash
dvc pull
```

To use wandb:

```bash
wandb login
```

```zsh
python run.py configs/exp_configs/inception_feature.yml configs/gan_configs/dcgan.yml
```


use pre-commit via 

```zsh
pre-commit run -a
```

## TODO:

* add Runner class to hold all needed inside and pass to all other objects
* CMD central moment discrepancy
  - do exps on images (ResNet)
* collect results in experiments forlder
* plan experiments and maintain actual list (+ push to dvc)
* stacked mnist
  - add model to count modes

  




