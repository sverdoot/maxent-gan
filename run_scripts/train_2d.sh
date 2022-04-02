#!/usr/bin/env bash

# # dumb

# python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.0001 --suffix wass_0.0001

# python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.0001 --suffix js_0.0001



python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v1_0.001 --feature_version 1
