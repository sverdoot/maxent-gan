# Maximum Entropy sampling for GANs revival

## 29/06/22

Maximum Entropy sampling with ULA and DumbFeature on DCGAN from ```torch_mimicry``` trained on CIFAR10 for 100k iterations.

```bash
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml configs/feature_configs/dumb.yml configs/common.yml --suffix ula
```

Results saved to ```log/inception_feature_DiscriminatorTarget/mmc_dcgan_ula```.

![mmc_dcgan_dumb_ula](../log/dumb_feature_DiscriminatorTarget/mmc_dcgan_ula/figs/fid.png)


Maximum Entropy sampling with ULA and InceptionFeature on DCGAN from ```torch_mimicry``` trained on CIFAR10 for 100k iterations.

```bash
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/exp_configs/mmc-dcgan-inception.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml configs/feature_configs/inception.yml configs/common.yml --suffix ula
```

Results saved to ```log/inception_feature_DiscriminatorTarget/mmc_dcgan_ula```.

![mmc_dcgan_inception_ula](../log/inception_feature_DiscriminatorTarget/mmc_dcgan_ula/figs/fid.png)

