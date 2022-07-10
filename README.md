# Experiments on sampling from GAN with probabilistic constraints

- [Experiments on sampling from GAN with probabilistic constraints](#experiments-on-sampling-from-gan-with-probabilistic-constraints)
  - [Installation](#installation)
    - [StudioGAN:](#studiogan)
  - [Usage](#usage)
  - [Tutorials](#tutorials)
  - [Results](#results)

## Installation

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
chmod +x -R scripts/*.sh
```

## Usage

Download checkpoints:

```bash
dvc pull
```

To use wandb:

```bash
wandb login
```

```zsh
    python run.py configs/exp_configs/dcgan-inception.yml configs/targets/discriminator.yml \
        configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/inception.yml \
        configs/mcmc_configs/{ula/mala/isir/ex2mcmc/flex2mcmc}.yml configs/mcmc_exp.yml
``` 


## Tutorials

* [docs/maxent_sample_tutorial.md](docs/maxent_sample_tutorial.md), [notebooks/maxent_sample_tutorial.ipynb](notebooks/maxent_sample_tutorial.ipynb)

* [docs/maxent_train_tutorial.md](docs/maxent_train_tutorial.md), [notebooks/maxent_train_tutorial.ipynb](../notebooks/maxent_train_tutorial.ipynb)


## Results







