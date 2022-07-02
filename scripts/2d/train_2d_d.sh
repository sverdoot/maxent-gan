# ring-js

#python train.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/dumb.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44

# # cond
# python train.py configs/exp_configs/mlp-dumb.yml configs/targets/conditional.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/dumb.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --feature_version 0 --seed 44

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_v1_0.1 --feature_version 1 --seed 44

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_v2_0.1 --feature_version 2 --seed 44

# python train.py configs/exp_configs/mlp-cmd.yml configs/targets/conditional.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/cmd_2d.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44

# python train.py configs/exp_configs/mlp-mmd.yml configs/targets/conditional.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44

# dis
# python train.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/dumb.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44

#python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.0001 --step_size 0.1 --suffix ring_d_0.1 --feature_version 0 --seed 44 --sweet_init

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_v1_0.1 --feature_version 1 --seed 44

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_v2_0.1 --feature_version 2 --seed 44

# python train.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/cmd_2d.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44

# python train.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44

# prior
python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.0001 --step_size 0.1 --suffix ring_d_0.1 --feature_version 0 --seed 44 --sweet_init

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_v1_0.1 --feature_version 1 --seed 44

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/ring_cluster.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_v2_0.1 --feature_version 2 --seed 44

# python train.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/cmd_2d.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44

# python train.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/ring-mlp.yml  configs/feature_configs/mmd_2d.yml configs/train_2d.yml --weight_step 0.1 --step_size 0.1 --suffix ring_d_0.1 --seed 44
