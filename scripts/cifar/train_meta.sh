#!/usr/bin/env bash

python train_meta.py configs/exp_configs/dcgan-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --weight_step 0.01 --suffix meta --seed 44