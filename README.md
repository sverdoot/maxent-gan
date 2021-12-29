# Experiments on sampling from GAN with constraints


- [Experiments on sampling from GAN with constraints](#experiments-on-sampling-from-gan-with-constraints)
  - [Getting started](#getting-started)
    - [StudioGAN:](#studiogan)
  - [Usage](#usage)
  - [TODO:](#todo)


link to gdrive with checkpoints and stats: https://drive.google.com/drive/folders/1bgIBVt5JAqb8RsHHDFRVp003YYjBoSXV?usp=sharing

## Getting started

Create environment and set dependencies:
```zsh
conda create -n constrained_gan python=3.8
conda activate constrained_gan
```

To compute FID in TF fashion:

```zsh
conda install tensorflow-gpu
```

```zsh
pip install poetry
poetry config virtualenvs.create false --local
```


### StudioGAN:

```zsh
git clone https://github.com/POSTECH-CVLab/PyTorch-StudioGAN.git thirdparty/studiogan && mv thirdparty/studiogan/src thirdparty/studiogan/studiogan
```


```zsh
touch thirdparty/studiogan/studiogan/__init__.py

echo "import sys

sys.path.append('.')
from . import config, utils" >> thirdparty/studiogan/studiogan/__init__.py

echo \
"from setuptools import setup, find_packages

setup(name='studiogan',
      version='1.0',
      packages=find_packages())" \
>> thirdparty/studiogan/setup.py
```

```zsh
pip install -e thirdparty/studiogan
```

```zsh
echo "tqdm ninja h5py kornia matplotlib pandas sklearn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg prdc" >> thirdparty/studiogan/requirements.txt
```

```zsh 
poetry add `cat thirdparty/studiogan/requirements.txt`
```

To use StudioGAN as is following is needed:

```python
import sys
from pathlib import Path

from soul_gan.utils.general_utils import ROOT_DIR

sys.path.append(Path(ROOT_DIR, 'thirdparty/studiogan/studiogan'))
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
chmod +x run_scripts/*.sh
```

## Usage 

Download checkpoints:

```bash
./run_scripts/get_ckpts.sh
./run_scripts/get_stats.sh
```

To use wandb:

```
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
* rewrite loop in run (still, looks weird)
* add more features and test
* add parallelism...
* make configs (gan_configs, not exp_configs (?)) for both freezed and non-freezed batch-norms


