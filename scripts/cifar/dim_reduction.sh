#!/usr/bin/env bash

declare -a METHODS=(
    "pca"
    "tsne"
    "umap"
    )


# for method in ${METHODS[@]}
# do
#     echo "$method"
#     python maxent_gan/utils/dim_reduction.py --method ${method} --train
# done

for method in ${METHODS[@]}
do
    echo "$method"
    python maxent_gan/utils/dim_reduction.py --method ${method} \
    --images \
        log/dumb_feature/dcgan_lips/images/0.npy \
        log/dumb_feature/dcgan_lips/images/500.npy \
        log/dumb_feature/wgan_gp_lips/images/0.npy \
        log/dumb_feature/wgan_gp_lips/images/500.npy \
        log/dumb_feature/sngan_ns_lips/images/0.npy \
        log/dumb_feature/sngan_ns_lips/images/500.npy \
        log/dumb_feature/snresnet_lips/images/0.npy \
        log/dumb_feature/snresnet_lips/images/500.npy \
        log/dumb_feature/studio_dcgan_lips/images/0.npy \
        log/dumb_feature/studio_dcgan_lips/images/500.npy \
        log/dumb_feature/studio_sngan_lips/images/0.npy \
        log/dumb_feature/studio_sngan_lips/images/500.npy \
        log/dumb_feature/studio_sagan_lips/images/0.npy \
        log/dumb_feature/studio_sagan_lips/images/500.npy \
        log/dumb_feature/studio_wgan_gp_lips/images/0.npy \
        log/dumb_feature/studio_wgan_gp_lips/images/300.npy \
    --save_dirs \
        log/dumb_feature/dcgan_lips/dim_reduction \
        log/dumb_feature/dcgan_lips/dim_reduction \
        log/dumb_feature/wgan_gp_lips/dim_reduction \
        log/dumb_feature/wgan_gp_lips/dim_reduction \
        log/dumb_feature/sngan_ns_lips/dim_reduction \
        log/dumb_feature/sngan_ns_lips/dim_reduction \
        log/dumb_feature/snresnet_lips/dim_reduction \
        log/dumb_feature/snresnet_lips/dim_reduction \
        log/dumb_feature/studio_dcgan_lips/dim_reduction \
        log/dumb_feature/studio_dcgan_lips/dim_reduction \
        log/dumb_feature/studio_sngan_lips/dim_reduction \
        log/dumb_feature/studio_sngan_lips/dim_reduction \
        log/dumb_feature/studio_sagan_lips/dim_reduction \
        log/dumb_feature/studio_sagan_lips/dim_reduction \
        log/dumb_feature/studio_wgan_gp_lips/dim_reduction \
        log/dumb_feature/studio_wgan_gp_lips/dim_reduction
done