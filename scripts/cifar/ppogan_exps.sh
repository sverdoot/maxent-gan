#!/usr/bin/env bash

python run.py configs/exp_configs/ppogan-dumb.yml \ configs/targets/discriminator.yml \
                configs/gan_configs/cifar-10-ppogan.yml  \configs/feature_configs/dumb.yml \ configs/mcmc_configs/ula.yml configs/mcmc_exp.yml --seed 0
