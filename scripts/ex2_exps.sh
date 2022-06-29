#!/usr/bin/env bash

# python run.py configs/exp_configs/sngan-ns-dumb-ex2.yml configs/mcmc_configs/grad_descent.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix gd

# python run.py configs/exp_configs/sngan-ns-dumb-ex2.yml configs/mcmc_configs/ula.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix ula

# python run.py configs/exp_configs/sngan-ns-dumb-ex2.yml configs/mcmc_configs/mala.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix mala

# python run.py configs/exp_configs/sngan-ns-dumb-ex2.yml configs/mcmc_configs/isir.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix isir

# python run.py configs/exp_configs/sngan-ns-dumb-ex2.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix ex2mcmc


# python run.py configs/exp_configs/snresnet-dumb-ex2.yml configs/mcmc_configs/grad_descent.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix gd

# python run.py configs/exp_configs/snresnet-dumb-ex2.yml configs/mcmc_configs/ula.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix ula

# python run.py configs/exp_configs/snresnet-dumb-ex2.yml configs/mcmc_configs/mala.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix mala

python run.py configs/exp_configs/snresnet-dumb-ex2.yml configs/mcmc_configs/isir.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix isir

python run.py configs/exp_configs/snresnet-dumb-ex2.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix ex2mcmc



# python run.py configs/exp_configs/wgan-gp-dumb-ex2.yml configs/mcmc_configs/ula.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix ula

# python run.py configs/exp_configs/wgan-gp-dumb-ex2.yml configs/mcmc_configs/grad_descent.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix gd

# python run.py configs/exp_configs/wgan-gp-dumb-ex2.yml configs/mcmc_configs/mala.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix mala

# python run.py configs/exp_configs/wgan-gp-dumb-ex2.yml configs/mcmc_configs/isir.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix isir

# python run.py configs/exp_configs/wgan-gp-dumb-ex2.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix ex2mcmc

# python run.py configs/exp_configs/wgan-gp-dumb-ex2.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix e



# python run.py configs/exp_configs/mmc-sngan-dumb-ex2.yml configs/mcmc_configs/ula.yml configs/targets/discriminator.yml configs/gan_configs/celeba-mmc-sngan.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix ula

# python run.py configs/exp_configs/mmc-sngan-dumb-ex2.yml configs/mcmc_configs/grad_descent.yml configs/targets/discriminator.yml configs/gan_configs/celeba-mmc-sngan.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix gd

# python run.py configs/exp_configs/mmc-sngan-dumb-ex2.yml configs/mcmc_configs/mala.yml configs/targets/discriminator.yml configs/gan_configs/celeba-mmc-sngan.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix mala

# python run.py configs/exp_configs/mmc-sngan-dumb-ex2.yml configs/mcmc_configs/isir.yml configs/targets/discriminator.yml configs/gan_configs/celeba-mmc-sngan.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix isir

# python run.py configs/exp_configs/mmc-sngan-dumb-ex2.yml configs/mcmc_configs/isir.yml configs/targets/discriminator.yml configs/gan_configs/celeba-mmc-sngan.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix isir

# python run.py configs/exp_configs/mmc-sngan-dumb-ex2.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/celeba-mmc-sngan.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix ex2mcmc

# python run.py configs/exp_configs/mmc-sngan-dumb-ex2.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/celeba-mmc-sngan.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix ex2mcmc


# python run.py configs/exp_configs/dcgan-dumb-ex2.yml configs/mcmc_configs/ula.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix ula

# python run.py configs/exp_configs/dcgan-dumb-ex2.yml configs/mcmc_configs/grad_descent.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix gd

# python run.py configs/exp_configs/dcgan-dumb-ex2.yml configs/mcmc_configs/mala.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/common_ex2.yml --step_size 0.025 --suffix mala

# python run.py configs/exp_configs/dcgan-dumb-ex2.yml configs/mcmc_configs/isir.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix isir

# python run.py configs/exp_configs/dcgan-dumb-ex2.yml configs/mcmc_configs/ex2mcmc.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/fid.yml --step_size 0.025 --suffix ex2mcmc
