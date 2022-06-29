#!/usr/bin/env bash

# DISCRIMINATOR 0.003

# # dumb

python run.py configs/exp_configs/dcgan-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003

# # discriminator 

#python run.py configs/exp_configs/dcgan-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/discriminator.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003

# # cluster 

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003 --feature_version 0

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix dis_0.003 --feature_version 0 --dis_emb

python run.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003 --feature_version 0


# # cluster_v1

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v1_0.003 --feature_version 1

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v1_dis_0.003 --feature_version 1 --dis_emb

python run.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v1_0.003 --feature_version 1


# # cluster_v2

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v2_0.003 --feature_version 2

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v2_dis_0.003 --feature_version 2 --dis_emb

# python run.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v2_0.003 --feature_version 2

# # mmd

# python run.py configs/exp_configs/dcgan-mmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003

#python run.py configs/exp_configs/dcgan-mmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix dis_0.003 --dis_emb

#python run.py configs/exp_configs/dcgan-mmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003


# # cmd

# python run.py configs/exp_configs/dcgan-cmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003

python run.py configs/exp_configs/dcgan-cmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix dis_0.003 --dis_emb

python run.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003






# PRIOR

# # discriminator 

#python run.py configs/exp_configs/dcgan-discriminator.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/discriminator.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003

# # cluster 

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003 --feature_version 0

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix dis_0.003 --feature_version 0 --dis_emb

# python run.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003 --feature_version 0


# # cluster_v1

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v1_0.003 --feature_version 1

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v1_dis_0.003 --feature_version 1 --dis_emb

# python run.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v1_0.003 --feature_version 1


# # cluster_v2

#python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v2_0.003 --feature_version 2

python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v2_dis_0.003 --feature_version 2 --dis_emb

#python run.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix v2_0.003 --feature_version 2

# # mmd

# python run.py configs/exp_configs/dcgan-mmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003

# python run.py configs/exp_configs/dcgan-mmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix dis_0.003 --dis_emb

# python run.py configs/exp_configs/dcgan-mmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003


# # cmd

#python run.py configs/exp_configs/dcgan-cmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003

python run.py configs/exp_configs/dcgan-cmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix dis_0.003 --dis_emb

python run.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd_resnet34.yml configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003



SCRIPT_PATH="./scripts/wgan-gp.sh"

# Here you execute your script
"$SCRIPT_PATH"


SCRIPT_PATH="./scripts/sngan-ns.sh"

# Here you execute your script
"$SCRIPT_PATH"


# SCRIPT_PATH="./scripts/snresnet.sh"

# # Here you execute your script
# "$SCRIPT_PATH"