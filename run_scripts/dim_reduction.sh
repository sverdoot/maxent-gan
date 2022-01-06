#!/usr/bin/bash
declare -a METHODS=(
    "pca"
    "tsne"
    "umap"
    )


for method in ${METHODS[@]}
do
    echo "$method"
    python dim_reduction.py --method ${method} --train
done

for method in ${METHODS[@]}
do
    echo "$method"
    python dim_reduction.py --method ${method} \
    --images \
        log/dumb_feature/dcgan/images/0.npy \
        log/dumb_feature/dcgan/images/500.npy \
        log/dumb_feature/wgan_gp/images/0.npy \
        log/dumb_feature/wgan_gp/images/500.npy \
        log/dumb_feature/sngan_ns/images/0.npy \
        log/dumb_feature/sngan_ns/images/600.npy \
        log/dumb_feature/snresnet/images/0.npy \
        log/dumb_feature/snresnet/images/500.npy \
        log/dumb_feature/studio_dcgan/images/0.npy \
        log/dumb_feature/studio_dcgan/images/500.npy \
        log/dumb_feature/studio_sngan/images/0.npy \
        log/dumb_feature/studio_sngan/images/600.npy \
        log/dumb_feature/studio_sagan/images/0.npy \
        log/dumb_feature/studio_sagan/images/600.npy \
        log/dumb_feature/studio_wgan_gp/images/0.npy \
        log/dumb_feature/studio_wgan_gp/images/300.npy \
    --save_dirs \
        log/dumb_feature/dcgan/dim_reduction \
        log/dumb_feature/dcgan/dim_reduction \
        log/dumb_feature/wgan_gp/dim_reduction \
        log/dumb_feature/wgan_gp/dim_reduction \
        log/dumb_feature/sngan_ns/dim_reduction \
        log/dumb_feature/sngan_ns/dim_reduction \
        log/dumb_feature/snresnet/dim_reduction \
        log/dumb_feature/snresnet/dim_reduction \
        log/dumb_feature/studio_dcgan/dim_reduction \
        log/dumb_feature/studio_dcgan/dim_reduction \
        log/dumb_feature/studio_sngan/dim_reduction \
        log/dumb_feature/studio_sngan/dim_reduction \
        log/dumb_feature/studio_sagan/dim_reduction \
        log/dumb_feature/studio_sagan/dim_reduction \
        log/dumb_feature/studio_wgan_gp/dim_reduction \
        log/dumb_feature/studio_wgan_gp/dim_reduction
done