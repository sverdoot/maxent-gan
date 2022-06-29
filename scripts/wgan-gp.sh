#!/usr/bin/env bash

# DISCRIMINATOR 0.001

# # dumb

#python run.py configs/exp_configs/wgan-gp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/dumb.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001

# # discriminator 

#python run.py configs/exp_configs/wgan-gp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/discriminator.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001

# # cluster 

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001 --feature_version 0

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix dis_0.001 --feature_version 0 --dis_emb

#python run.py configs/exp_configs/wgan-gp-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001 --feature_version 0


# # cluster_v1

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v1_0.001 --feature_version 1

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v1_dis_0.001 --feature_version 1 --dis_emb

#python run.py configs/exp_configs/wgan-gp-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v1_0.001 --feature_version 1


# # cluster_v2

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v2_0.001 --feature_version 2

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v2_dis_0.001 --feature_version 2 --dis_emb

# python run.py configs/exp_configs/wgan-gp-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v2_0.001 --feature_version 2

# # mmd

# python run.py configs/exp_configs/wgan-gp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001

#python run.py configs/exp_configs/wgan-gp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix dis_0.001 --dis_emb

#python run.py configs/exp_configs/wgan-gp-mmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/mmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001


# # cmd

# python run.py configs/exp_configs/wgan-gp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001

#python run.py configs/exp_configs/wgan-gp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix dis_0.001 --dis_emb

python run.py configs/exp_configs/wgan-gp-cmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001






# PRIOR

# # discriminator 

#python run.py configs/exp_configs/wgan-gp-discriminator.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/discriminator.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001

# # cluster 

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001 --feature_version 0

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix dis_0.001 --feature_version 0 --dis_emb

# python run.py configs/exp_configs/wgan-gp-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001 --feature_version 0


# # cluster_v1

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v1_0.001 --feature_version 1

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v1_dis_0.001 --feature_version 1 --dis_emb

# python run.py configs/exp_configs/wgan-gp-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v1_0.001 --feature_version 1


# # cluster_v2

#python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v2_0.001 --feature_version 2

python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v2_dis_0.001 --feature_version 2 --dis_emb

#python run.py configs/exp_configs/wgan-gp-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix v2_0.001 --feature_version 2

# # mmd

# python run.py configs/exp_configs/wgan-gp-mmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001

# python run.py configs/exp_configs/wgan-gp-mmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix dis_0.001 --dis_emb

# python run.py configs/exp_configs/wgan-gp-mmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/mmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001


# # cmd

#python run.py configs/exp_configs/wgan-gp-cmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001

python run.py configs/exp_configs/wgan-gp-cmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix dis_0.001 --dis_emb

python run.py configs/exp_configs/wgan-gp-cmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001



python run.py configs/exp_configs/wgan-gp-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml configs/feature_configs/resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.001 --suffix 0.001


SCRIPT_PATH="./scripts/sngan-ns.sh"

# Here you execute your script
"$SCRIPT_PATH"