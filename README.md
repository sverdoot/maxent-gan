# Experiments on sampling from GAN with constraints


- [Experiments on sampling from GAN with constraints](#experiments-on-sampling-from-gan-with-constraints)
  - [Getting started](#getting-started)
  - [Usage](#usage)
  - [TODO:](#todo)




## Getting started

Create environment and set dependencies:
```zsh
conda create -n constrained_gan python=3.8
conda activate constrained_gan
```

```zsh
pip install poetry
poetry config virtualenvs.create false --local
```

```zsh
conda activate soul
```

```zsh
poetry install
```

To compute FID in TF fashion:

```zsh
wget  "https://raw.githubusercontent.com/bioinf-jku/TTUR/master/fid.py"  -P thirdparty/TTUR
```

```zsh
conda install tensorflow-gpu
```

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

* rewrite loop in run
* add more features and test
* add parallelism...


