# python run.py configs/exp_configs/dcgan-discriminator.yml configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/discriminator.yml --lipschitz_step_size --step_size_mul 0.25

# python run.py configs/exp_configs/wgan-gp-discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml configs/feature_configs/discriminator.yml --lipschitz_step_size --step_size_mul 0.25

# python run.py configs/exp_configs/sngan-ns-discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml configs/feature_configs/discriminator.yml --lipschitz_step_size  --step_size_mul 0.25

# python run.py configs/exp_configs/snresnet-discriminator.yml configs/gan_configs/cifar-10-snresnet.yml configs/feature_configs/discriminator.yml --lipschitz_step_size  --step_size_mul 0.25

# dis
# done python run.py configs/exp_configs/dcgan-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.1

# done python run.py configs/exp_configs/dcgan-discriminator.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.1

# done python run.py configs/exp_configs/dcgan-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --step_size 0.001 --suffix 0.001 --weight_step 0.1

# done python run.py configs/exp_configs/wgan-gp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.1

# python run.py configs/exp_configs/wgan-gp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --step_size 0.001 --suffix 0.001 --weight_step 0.1

# done python run.py configs/exp_configs/sngan-ns-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.1

# todo python run.py configs/exp_configs/snresnet-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-snresnet.yml configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.01

# done python run.py configs/exp_configs/studio-dcgan-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-studio-dcgan.yml configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.1


# cluster
# done python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 1.0 #0.1

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 1.0

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 1.0

# # python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/feature_configs/common.yml --step_size 0.001 --suffix 0.001 --weight_step 1.0

# #done python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 1.0

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 1.0

# # python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/feature_configs/common.yml --step_size 0.001 --suffix 0.001 --weight_step 1.0

# python run.py configs/exp_configs/sngan-ns-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml configs/feature_configs/cluster.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 1.0

# python run.py configs/exp_configs/studio-dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-studio-dcgan.yml configs/feature_configs/cluster.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 1.0


# # pca
# python run.py configs/exp_configs/dcgan-pca.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/pca.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 1.0 #0.1

# # python run.py configs/exp_configs/dcgan-pca.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/pca.yml configs/feature_configs/common.yml --step_size 0.001 --suffix 0.001 --weight_step 1.0

# python run.py configs/exp_configs/wgan-gp-pca.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/pca.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 1.0

# # python run.py configs/exp_configs/wgan-gp-pca.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/pca.yml configs/feature_configs/common.yml --step_size 0.001 --suffix 0.001 --weight_step 1.0

# # resnet34
# done python run.py configs/exp_configs/dcgan-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/resnet34.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.5

# done python run.py configs/exp_configs/wgan-gp-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/resnet34.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.1

# done python run.py configs/exp_configs/sngan-ns-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/resnet34.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01 --weight_step 0.1

# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 1.0




# python run.py configs/exp_configs/dcgan-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/discriminator_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 0.1

# python run.py configs/exp_configs/wgan-gp-discriminator.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml configs/feature_configs/discriminator_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 0.1 #0001

#python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 0.01 #1.0

python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-dcgan.yml configs/feature_configs/cluster_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 0.3 #3

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml configs/feature_configs/cluster_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 0.1

# python run.py configs/exp_configs/sngan-ns-cluster.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml configs/feature_configs/cluster_v1.yml configs/feature_configs/common.yml --step_size 0.01 --suffix v1_0.01 --weight_step 0.1