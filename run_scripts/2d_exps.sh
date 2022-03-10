#!/usr/bin/env bash

# dumb

python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.001 --suffix wass_0.001

python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.001 --suffix js_0.001

python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.001 --suffix ring_0.001


# discriminator 

python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix wass_0.001

python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix js_0.001

python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 0.1 --step_size 0.001 --suffix ring_0.001


# cluster 

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.01 --suffix wass_0.01 --feature_version 0

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.01 --suffix js_0.01 --feature_version 0

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.01 --suffix ring_0.01 --feature_version 0


# cluster_v1

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.01 --suffix wass_v1_0.01 --feature_version 1

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.01 --suffix js_v1_0.01 --feature_version 1

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.01 --suffix ring_v1_0.01 --feature_version 1


# cluster_v2

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix wass_v2_0.001 --feature_version 2

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix js_v2_0.001 --feature_version 2

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v2_0.001 --feature_version 2

# cluster_v3

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix wass_v3_0.001 --feature_version 3

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix js_v3_0.001 --feature_version 3

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v3_0.001 --feature_version 3


