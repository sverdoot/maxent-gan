
#!/usr/bin/env bash

#python train.py configs/exp_configs/dcgan-dumb.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

# cond
# python train.py configs/exp_configs/dcgan-dumb.yml configs/targets/conditional.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

#python train.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/conditional.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster_resnet34.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --feature_version 0 --seed 44

# python train.py configs/exp_configs/dcgan-cluster.yml configs/targets/conditional.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix v1_0.01 --feature_version 1 --seed 44

# python train.py configs/exp_configs/dcgan-cluster.yml configs/targets/conditional.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix v2_0.01 --feature_version 2 --seed 44

#python train.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/conditional.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

#python train.py configs/exp_configs/dcgan-mmd-resnet34.yml configs/targets/conditional.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

# # dis

# python train.py configs/exp_configs/dcgan-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/dumb.yml configs/train.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

# python train.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --feature_version 0 --seed 44

# # python train.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix v1_0.01 --feature_version 1 --seed 44

# # python train.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix v2_0.01 --feature_version 2 --seed 44

# python train.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/train.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01_test --seed 44

#python train.py configs/exp_configs/dcgan-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/resnet34.yml configs/train.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

#python train.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd_resnet34.yml configs/train.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

python train.py configs/exp_configs/dcgan-cmd.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/train.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44


# python train.py configs/exp_configs/dcgan-mmd-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44


# prior
#python train.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --feature_version 0 --seed 44

# python train.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix v1_0.01 --feature_version 1 --seed 44

# python train.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix v2_0.01 --feature_version 2 --seed 44

# python train.py configs/exp_configs/dcgan-cmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/train.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

# python train.py configs/exp_configs/dcgan-cmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix iw_0.01 --seed 44


#python train.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd_resnet34.yml configs/train.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

#python train.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd_resnet34.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix iw_0.01 --seed 44  #--resume


# python train.py configs/exp_configs/dcgan-mmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/train.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

#python train.py configs/exp_configs/dcgan-mmd.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix iw_0.01 --seed 44


# python train.py configs/exp_configs/dcgan-cmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cmd_resnet34.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44

#python train.py configs/exp_configs/dcgan-mmd-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/mmd.yml configs/train_iw.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01 --seed 44
