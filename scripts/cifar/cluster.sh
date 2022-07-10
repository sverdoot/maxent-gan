#!/usr/bin/env bash

python maxent_gan/utils/cluster.py --n_clusters 100 --dataset cifar10

python maxent_gan/utils/cluster.py --n_clusters 100 --dataset celeba
