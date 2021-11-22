#!/usr/bin/env bash

for iter in {1..3}
do
    python run.py configs/exp_configs/dumb_feature.yml configs/gan_configs/dcgan.yml --seed 4$iter
done

for iter in {1..3}
do
    python run.py configs/exp_configs/discriminator_feature.yml configs/gan_configs/dcgan.yml --seed 4$iter
done

for iter in {1..3}
do
    python run.py configs/exp_configs/inception_feature.yml configs/gan_configs/dcgan.yml --seed 4$iter
done

for iter in {1..3}
do
    python run.py configs/exp_configs/inception_mean_feature.yml configs/gan_configs/dcgan.yml --seed 4$iter
done

