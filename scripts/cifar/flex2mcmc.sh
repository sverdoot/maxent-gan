#!/usr/bin/env bash

# python run.py configs/exp_configs/dcgan-inception.yml configs/targets/discriminator.yml \
#     configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/inception.yml \
#     "configs/mcmc_configs/flex2mcmc.yml" configs/mcmc_exp.yml --suffix flex --seed 0

python run.py configs/exp_configs/dcgan-dumb.yml configs/targets/discriminator.yml \
    configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/dumb.yml \
    "configs/mcmc_configs/flex2mcmc.yml" configs/mcmc_exp.yml --suffix flex --seed 0