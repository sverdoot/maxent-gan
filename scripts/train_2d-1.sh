#!/usr/bin/env bash


# # grid-js

# python train.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47

# # cond
# python train.py configs/exp_configs/mlp-dumb.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --feature_version 0 --seed 47

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v1_0.01 --feature_version 1 --seed 47

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v2_0.01 --feature_version 2 --seed 47

python train.py configs/exp_configs/mlp-cmd.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47

python train.py configs/exp_configs/mlp-mmd.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47

# dis
python train.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --feature_version 0 --seed 47

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v1_0.01 --feature_version 1 --seed 47

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v2_0.01 --feature_version 2 --seed 47

python train.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47

python train.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47

# prior
python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --feature_version 0 --seed 47

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v1_0.01 --feature_version 1 --seed 47

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v2_0.01 --feature_version 2 --seed 47

python train.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47

python train.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 47


# grid-js

python train.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46

# cond
python train.py configs/exp_configs/mlp-dumb.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --feature_version 0 --seed 46

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v1_0.01 --feature_version 1 --seed 46

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v2_0.01 --feature_version 2 --seed 46

python train.py configs/exp_configs/mlp-cmd.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46

python train.py configs/exp_configs/mlp-mmd.yml configs/targets/conditional.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46

# dis
python train.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --feature_version 0 --seed 46

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v1_0.01 --feature_version 1 --seed 46

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v2_0.01 --feature_version 2 --seed 46

python train.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46

python train.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46

# prior
python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --feature_version 0 --seed 46

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v1_0.01 --feature_version 1 --seed 46

python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_v2_0.01 --feature_version 2 --seed 46

python train.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46

python train.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-js-mlp2.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix js_0.01 --seed 46





# # # grid-wass

# python train.py configs/exp_configs/mlp-dumb.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01

# # cond
# python train.py configs/exp_configs/mlp-dumb.yml configs/targets/conditional.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01 --feature_version 0

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_v1_0.01 --feature_version 1

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/conditional.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_v2_0.01 --feature_version 2

# python train.py configs/exp_configs/mlp-cmd.yml configs/targets/conditional.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01

# python train.py configs/exp_configs/mlp-mmd.yml configs/targets/conditional.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01

# # dis
# python train.py configs/exp_configs/mlp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/dumb.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01 --feature_version 0

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_v1_0.01 --feature_version 1

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_v2_0.01 --feature_version 2

# python train.py configs/exp_configs/mlp-cmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01

# python train.py configs/exp_configs/mlp-mmd.yml configs/targets/discriminator.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01

# # prior
# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01 --feature_version 0

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_v1_0.01 --feature_version 1

# python train.py configs/exp_configs/mlp-cluster.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/grid_cluster.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_v2_0.01 --feature_version 2

# python train.py configs/exp_configs/mlp-cmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/cmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01

# python train.py configs/exp_configs/mlp-mmd.yml configs/targets/prior.yml configs/gan_configs/grid-wass-mlp.yml  configs/feature_configs/mmd_2d.yml configs/train_2d_iw.yml --weight_step 0.1 --step_size 0.01 --suffix wass_0.01


