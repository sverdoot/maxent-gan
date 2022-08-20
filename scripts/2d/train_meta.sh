#!/usr/bin/env bash

# ring-2d
# python train_meta.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta_2d.yml --suffix ring_meta --seed 44 --sample_steps 0 --step_size 0.01 --step_every 0

# python train_meta.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta_2d.yml --suffix ring_meta_1 --seed 44 --sample_steps 1 --step_size 0.01 --step_every 1

# python train_meta.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/cmd_2d.yml configs/mcmc_configs/ula.yml configs/train_meta_2d.yml --weight_step 0.01 --suffix ring_meta_1 --seed 44 --sample_steps 1 --step_size 0.01 --step_every 1


# grid-2d
# python train_meta.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta_2d.yml --suffix grid_meta --seed 44 --sample_steps 0 --step_size 0.01

# python train_meta.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta_2d.yml --suffix grid_meta_1 --seed 44 --sample_steps 1 --step_size 0.01

# python train_meta.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/cmd_2d.yml configs/mcmc_configs/ula.yml configs/train_meta_2d.yml --weight_step 0.01 --suffix grid_meta_1 --seed 44 --sample_steps 1 --step_size 0.01


# # # grid-3d
# # python train_meta.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/grid3-js-mlp.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta_3d.yml --suffix 3d_meta --seed 44 --sample_steps 0 --step_size 0.01

# # python train_meta.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid3-js-mlp.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta_3d.yml --suffix 3d_meta_1 --seed 44 --sample_steps 1 --step_size 0.01

# # python train_meta.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid3-js-mlp.yml  configs/feature_configs/cmd_2d.yml configs/mcmc_configs/ula.yml configs/train_meta_3d.yml --weight_step 0.01 --suffix 3d_meta_1 --seed 44 --sample_steps 1 --step_size 0.01


# # cifar10
python train_meta.py configs/exp_configs/dcgan-dumb.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --suffix meta --seed 44 --sample_steps 0 --step_size 0.01

python train_meta.py configs/exp_configs/dcgan-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --suffix meta_1 --seed 44 --sample_steps 1 --step_size 0.01

python train_meta.py configs/exp_configs/dcgan-inception.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/inception.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --weight_step 0.01 --suffix meta_1 --seed 44 --sample_steps 1 --step_size 0.01

python train_meta.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd_resnet34.yml configs/mcmc_configs/ula.yml configs/train_meta.yml --weight_step 0.01 --suffix meta_1 --seed 44 --sample_steps 1 --step_size 0.01
