# python run.py configs/exp_configs/dcgan-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster.yml configs/feature_configs/common.yml

# python run.py configs/exp_configs/wgan-gp-cluster.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster.yml configs/feature_configs/common.yml


# python run.py configs/exp_configs/dcgan-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/cluster_resnet34.yml configs/feature_configs/common.yml --weight_step 0.1

# python run.py configs/exp_configs/wgan-gp-cluster-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster_resnet34.yml configs/feature_configs/common.yml --weight_step 0.1


# python run.py configs/exp_configs/wgan-gp-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/resnet34.yml configs/feature_configs/common.yml --weight_step 0.1



#python run.py configs/exp_configs/wgan-gp-cluster-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/cluster_resnet34.yml configs/feature_configs/common.yml --weight_step 1


# python run.py configs/exp_configs/dcgan-discriminator.yml configs/targets/prior.yml configs/gan_configs/cifar-10-dcgan.yml  configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --weight_step 0.01

# python run.py configs/exp_configs/wgan-gp-discriminator.yml configs/targets/prior.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/discriminator.yml configs/feature_configs/common.yml --weight_step 0.01


#python run.py configs/exp_configs/wgan-gp-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/resnet34.yml configs/feature_configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.01

#python run.py configs/exp_configs/wgan-gp-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-wgan-gp.yml  configs/feature_configs/dumb.yml configs/feature_configs/common.yml --step_size 0.01 --suffix 0.01



# sngan-ns

python run.py configs/exp_configs/sngan-ns-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/dumb.yml configs/feature_configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003
python run.py configs/exp_configs/sngan-ns-resnet34.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/resnet34.yml configs/feature_configs/common.yml --weight_step 0.1 --step_size 0.003 --suffix 0.003
# python run.py configs/exp_configs/sngan-ns-resnet34.yml configs/targets/prior.yml configs/gan_configs/cifar-10-sngan-ns.yml  configs/feature_configs/resnet34.yml configs/feature_configs/common.yml --weight_step 0.1 --step_size 0.01 --suffix 0.001
