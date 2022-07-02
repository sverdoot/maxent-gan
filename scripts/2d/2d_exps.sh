#!/usr/bin/env bash

# DISCRIMINATOR

# # dumb

# python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.0001 --suffix wass_0.0001

# python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.0001 --suffix js_0.0001

python run.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.001 --suffix ring_0.001


# # # discriminator 

# python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_0.0001

# python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_0.0001

python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001


# # # cluster 

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_0.0001 --feature_version 0

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_0.0001 --feature_version 0

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001 --feature_version 0


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_dis_0.0001 --feature_version 0 --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_dis_0.0001 --feature_version 0 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_dis_0.001 --feature_version 0 --dis_emb


# # cluster_v1

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v1_0.0001 --feature_version 1

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v1_0.0001 --feature_version 1

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v1_0.001 --feature_version 1


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v1_dis_0.0001 --feature_version 1 --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v1_dis_0.0001 --feature_version 1 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v1_dis_0.001 --feature_version 1 --dis_emb


# # cluster_v2

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_0.0001 --feature_version 2

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_0.0001 --feature_version 2

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v2_0.001 --feature_version 2


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_lin_0.0001 --feature_version 2 --kernel LinearKernel

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_lin_0.0001 --feature_version 2 --kernel LinearKernel

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_v2_lin_0.0001 --feature_version 2 --kernel LinearKernel


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_quad_0.0001 --feature_version 2 --kernel PolynomialKernel

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_quad_0.0001 --feature_version 2 --kernel PolynomialKernel

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_v2_quad_0.0001 --feature_version 2 --kernel PolynomialKernel



# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_dis_0.0001 --feature_version 2 --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_dis_0.0001 --feature_version 2 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v2_dis_0.001 --feature_version 2 --dis_emb


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_lin_dis_0.0001 --feature_version 2 --kernel LinearKernel --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_lin_dis_0.0001 --feature_version 2 --kernel LinearKernel --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_v2_lin_dis_0.0001 --feature_version 2 --kernel LinearKernel --dis_emb


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_quad_dis_0.0001 --feature_version 2 --kernel PolynomialKernel --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_quad_dis_0.0001 --feature_version 2 --kernel PolynomialKernel --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_v2_quad_dis_0.0001 --feature_version 2 --kernel PolynomialKernel --dis_emb


# # cluster_v3

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v3_0.0001 --feature_version 3

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v3_0.0001 --feature_version 3

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v3_0.001 --feature_version 3


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v3_dis_0.0001 --feature_version 3 --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v3_dis_0.0001 --feature_version 3 --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v3_dis_0.001 --feature_version 3 --dis_emb


# # mmd

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_0.0001

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_0.0001

python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001


# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_lin_0.0001 --kernel LinearKernel

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_lin_0.0001 --kernel LinearKernel

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_lin_0.0001 --kernel LinearKernel


# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_quad_0.0001 --kernel PolynomialKernel

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_quad_0.0001 --kernel PolynomialKernel

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_quad_0.0001 --kernel PolynomialKernel



# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_dis_0.0001 --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_dis_0.0001 --dis_emb

python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_dis_0.001 --dis_emb


# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_lin_dis_0.0001 --kernel LinearKernel --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_lin_dis_0.0001 --kernel LinearKernel --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_lin_dis_0.0001 --kernel LinearKernel --dis_emb


# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_quad_dis_0.0001 --kernel PolynomialKernel --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_quad_dis_0.0001 --kernel PolynomialKernel --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_quad_dis_0.0001 --kernel PolynomialKernel --dis_emb


# # cmd

# python run.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_0.0001

# python run.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_0.0001

python run.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001


# python run.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_dis_0.0001 --dis_emb

# python run.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_dis_0.0001 --dis_emb

python run.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_dis_0.001 --dis_emb


# PRIOR

# dumb

python run.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.0001 --suffix wass_0.0001

python run.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.0001 --suffix js_0.0001

python run.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/dumb.yml configs/common_2d.yml --weight_step 0.0 --step_size 0.001 --suffix ring_0.001



# # discriminator 

python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_0.0001

python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_0.0001

python run.py configs/exp_configs/mlp-discriminator.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/discriminator.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001


# # cluster 

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_0.0001 --feature_version 0

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_0.0001 --feature_version 0

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001 --feature_version 0


python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_dis_0.0001 --feature_version 0 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_dis_0.0001 --feature_version 0 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_dis_0.001 --feature_version 0 --dis_emb


# cluster_v1

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v1_0.0001 --feature_version 1

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v1_0.0001 --feature_version 1

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v1_0.001 --feature_version 1


python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v1_dis_0.0001 --feature_version 1 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v1_dis_0.0001 --feature_version 1 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v1_dis_0.001 --feature_version 1 --dis_emb


# cluster_v2

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_0.0001 --feature_version 2

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_0.0001 --feature_version 2

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v2_0.001 --feature_version 2


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_lin_0.0001 --feature_version 2 --kernel LinearKernel

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_lin_0.0001 --feature_version 2 --kernel LinearKernel

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_v2_lin_0.0001 --feature_version 2 --kernel LinearKernel


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_quad_0.0001 --feature_version 2 --kernel PolynomialKernel

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_quad_0.0001 --feature_version 2 --kernel PolynomialKernel

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_v2_quad_0.0001 --feature_version 2 --kernel PolynomialKernel



python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_dis_0.0001 --feature_version 2 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_dis_0.0001 --feature_version 2 --dis_emb

python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v2_dis_0.001 --feature_version 2 --dis_emb


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_lin_dis_0.0001 --feature_version 2 --kernel LinearKernel --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_lin_dis_0.0001 --feature_version 2 --kernel LinearKernel --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_v2_lin_dis_0.0001 --feature_version 2 --kernel LinearKernel --dis_emb


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v2_quad_dis_0.0001 --feature_version 2 --kernel PolynomialKernel --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v2_quad_dis_0.0001 --feature_version 2 --kernel PolynomialKernel --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_v2_quad_dis_0.0001 --feature_version 2 --kernel PolynomialKernel --dis_emb


# # cluster_v3

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v3_0.0001 --feature_version 3

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v3_0.0001 --feature_version 3

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v3_0.001 --feature_version 3


# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_v3_dis_0.0001 --feature_version 3 --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/grid_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_v3_dis_0.0001 --feature_version 3 --dis_emb

# python run.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_v3_dis_0.001 --feature_version 3 --dis_emb


# mmd

python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_0.0001

python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_0.0001

python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001


# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_lin_0.0001 --kernel LinearKernel

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_lin_0.0001 --kernel LinearKernel

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_lin_0.0001 --kernel LinearKernel


# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_quad_0.0001 --kernel PolynomialKernel

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_quad_0.0001 --kernel PolynomialKernel

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_quad_0.0001 --kernel PolynomialKernel



python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_dis_0.0001 --dis_emb

python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_dis_0.0001 --dis_emb

python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_dis_0.001 --dis_emb


# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_lin_dis_0.0001 --kernel LinearKernel --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_lin_dis_0.0001 --kernel LinearKernel --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_lin_dis_0.0001 --kernel LinearKernel --dis_emb


# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_quad_dis_0.0001 --kernel PolynomialKernel --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_quad_dis_0.0001 --kernel PolynomialKernel --dis_emb

# python run.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix ring_quad_dis_0.0001 --kernel PolynomialKernel --dis_emb


# cmd

python run.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_0.0001

python run.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_0.0001

python run.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_0.001


python run.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix wass_dis_0.0001 --dis_emb

python run.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.0001 --suffix js_dis_0.0001 --dis_emb

python run.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/cmd_2d.yml configs/common_2d.yml --weight_step 1.0 --step_size 0.001 --suffix ring_dis_0.001 --dis_emb


# SCRIPT_PATH="./scripts/train_3d.sh"

# # Here you execute your script
# "$SCRIPT_PATH"