# Experiments on sampling from GAN with constraints

## Getting started

```zsh
conda create -n constrained_gan python=3.8
conda activate constrained_gan
```

```zsh
pip install poetry
poetry config virtualenvs.create false --local
```

```zsh
poetry install
```

```zsh
conda activate soul
```

Put CIFAR-10 into directory ```data/cifar10```  using this script

```python
import torchvision.datasets as dset

cifar = dset.CIFAR10(root='data/cifar10', download=True)
```


## Usage 


```zsh
python run.py configs/exp_configs/inception_feature.yml configs/gan_configs/dcgan.yml
```




