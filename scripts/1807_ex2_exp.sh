#!/usr/bin/env bash

python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/fid.yml --suffix 1807

# python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/fid.yml --suffix 1807

python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/mala.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/fid.yml --suffix 1807

python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/isir.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/fid.yml --suffix 1807


# python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/flex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_exp.yml --suffix 1807



# python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_exp.yml --suffix 1807_1 --seed 1

# # python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/fid.yml --suffix 1807

# python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/mala.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_exp.yml --suffix 1807_1 --seed 1

# python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/mcmc_configs/isir.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_exp.yml --suffix 1807_1 --seed 1
