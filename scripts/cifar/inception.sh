#!/usr/bin/env bash


for method in mala #isir ex2mcmc
do
   
    python run.py configs/exp_configs/dcgan-inception.yml configs/targets/discriminator.yml \
        configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/inception.yml \
        "configs/mcmc_configs/${method}.yml" configs/mcmc_exp.yml --suffix ${method} --seed 0
    
    python run.py configs/exp_configs/dcgan-dumb.yml configs/targets/discriminator.yml \
        configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/dumb.yml \
        "configs/mcmc_configs/${method}.yml" configs/mcmc_exp.yml --suffix ${method} --seed 0

done


for method in mala #isir ex2mcmc
do
   
    python run.py configs/exp_configs/wgan-gp-in-inception.yml configs/targets/discriminator.yml \
        configs/gan_configs/cifar-10-wgan-gp-in.yml configs/feature_configs/inception.yml \
        "configs/mcmc_configs/${method}.yml" configs/mcmc_exp.yml --suffix ${method} --seed 0
    
    python run.py configs/exp_configs/wgan-gp-in-dumb.yml configs/targets/discriminator.yml \
        configs/gan_configs/cifar-10-wgan-gp-in.yml configs/feature_configs/dumb.yml \
        "configs/mcmc_configs/${method}.yml" configs/mcmc_exp.yml --suffix ${method} --seed 0

done


for method in mala #isir ex2mcmc
do
   
    python run.py configs/exp_configs/mmc-dcgan-inception.yml configs/targets/discriminator.yml \
        configs/gan_configs/cifar-10-mmc-dcgan.yml configs/feature_configs/inception.yml \
        "configs/mcmc_configs/${method}.yml" configs/mcmc_exp.yml --suffix ${method} --seed 0
    
    python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/targets/discriminator.yml \
        configs/gan_configs/cifar-10-mmc-dcgan.yml configs/feature_configs/dumb.yml \
        "configs/mcmc_configs/${method}.yml" configs/mcmc_exp.yml --suffix ${method} --seed 0

done
