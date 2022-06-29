#!/usr/bin/env bash

# DISCRIMINATOR

# # dumb

# python run.py configs/exp_configs/snresnet-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/dumb.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01

# # discriminator 

#python run.py configs/exp_configs/snresnet-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/discriminator.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01

# # cluster 

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --feature_version 0

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix dis_0.01 --feature_version 0 --dis_emb

# python run.py configs/exp_configs/snresnet-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/fid.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --feature_version 0


# # cluster_v1

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v1_0.01 --feature_version 1

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v1_dis_0.01 --feature_version 1 --dis_emb

# python run.py configs/exp_configs/snresnet-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v1_0.01 --feature_version 1


# # cluster_v2

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v2_0.01 --feature_version 2

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v2_dis_0.01 --feature_version 2 --dis_emb

# python run.py configs/exp_configs/snresnet-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v2_0.01 --feature_version 2

# # mmd

# python run.py configs/exp_configs/snresnet-mmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01

#python run.py configs/exp_configs/snresnet-mmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix dis_0.01 --dis_emb

#python run.py configs/exp_configs/snresnet-mmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/mmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01


# # cmd

# python run.py configs/exp_configs/snresnet-cmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01

# python run.py configs/exp_configs/snresnet-cmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix dis_0.01 --dis_emb

python run.py configs/exp_configs/snresnet-cmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cmd_resnet34.yml configs/fid.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01






# PRIOR

# # discriminator 

#python run.py configs/exp_configs/snresnet-discriminator.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/discriminator.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01

# # cluster 

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --feature_version 0

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix dis_0.01 --feature_version 0 --dis_emb

# python run.py configs/exp_configs/snresnet-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --feature_version 0


# # cluster_v1

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v1_0.01 --feature_version 1

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v1_dis_0.01 --feature_version 1 --dis_emb

# python run.py configs/exp_configs/snresnet-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v1_0.01 --feature_version 1


# # cluster_v2

#python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v2_0.01 --feature_version 2

# python run.py configs/exp_configs/snresnet-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/fid.yml --weight_step 0.1 --step_size 0.01 --suffix v2_dis_0.01 --feature_version 2 --dis_emb

#python run.py configs/exp_configs/snresnet-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix v2_0.01 --feature_version 2

# # mmd

# python run.py configs/exp_configs/snresnet-mmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01

# python run.py configs/exp_configs/snresnet-mmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix dis_0.01 --dis_emb

# python run.py configs/exp_configs/snresnet-mmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/mmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01


# # cmd

#python run.py configs/exp_configs/snresnet-cmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01

# python run.py configs/exp_configs/snresnet-cmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix dis_0.01 --dis_emb

python run.py configs/exp_configs/snresnet-cmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-snresnet.yml  configs/feature_configs/cmd_resnet34.yml configs/fid.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01
