#!/usr/bin/env bash

#python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/gauss2d-wass-mlp.yml  configs/feature_configs/dumb.yml configs/common_gauss.yml --weight_step 0.0 --step_size 0.001 --suffix wass_0.001

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/gauss2d-wass-mlp.yml  configs/feature_configs/gauss_cluster.yml configs/common_gauss.yml --weight_step 1.0 --step_size 0.001 --suffix wass_0.001

# #python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/gauss2d-js-mlp.yml  configs/feature_configs/dumb.yml configs/common_gauss.yml --weight_step 0.0 --step_size 0.001 --suffix js_0.001

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/gauss2d-js-mlp.yml  configs/feature_configs/gauss_cluster.yml configs/common_gauss.yml --weight_step 1.0 --step_size 0.001 --suffix js_0.001

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/gauss2d-wass-mlp.yml  configs/feature_configs/gauss_cluster.yml configs/common_gauss.yml --weight_step 1.0 --step_size 0.001 --suffix wass_0.001

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/gauss2d-js-mlp.yml  configs/feature_configs/gauss_cluster.yml configs/common_gauss.yml --weight_step 1.0 --step_size 0.001 --suffix js_0.001

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_gauss.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001
