#!/usr/bin/env bash

python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/gauss2d-mlp.yml  configs/feature_configs/dumb.yml configs/feature_configs/common_gauss.yml --weight_step 0.0 --step_size 0.1 --suffix 0.1

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/gauss2d-mlp.yml  configs/feature_configs/gauss_cluster.yml configs/feature_configs/common_gauss.yml --weight_step 1.0 --step_size 0.1 --suffix 0.1