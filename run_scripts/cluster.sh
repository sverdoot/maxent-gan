#!/usr/bin/env bash

python soul_gan/utils/cluster.py --n_clusters 100 --dataset cifar10

python soul_gan/utils/cluster.py --n_clusters 100 --dataset celeba

python soul_gan/utils/cluster.py --n_clusters 1000 --dataset stacked_mnist --norm_mean 0.5 --norm_std 0.5