#!/usr/bin/env bash

# dumb, meta
# python train_meta.py configs/exp_configs/dcgan-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --weight_step 0.01 --suffix meta --seed 44 --sample_steps 1

# vanilla
python train_meta.py configs/exp_configs/dcgan-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --weight_step 0.01 --seed 44 --sample_steps 0

# # inception, meta
# python train_meta.py configs/exp_configs/dcgan-inception.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/inception.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --weight_step 0.01 --suffix meta --seed 44 --sample_steps 1


# # #inception, meta
# #python train_meta.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --weight_step 0.01 --suffix meta --seed 44 --sample_steps 1
