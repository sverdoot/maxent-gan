# Maximum Entropy sampling for GANs revival

## 29/06/22

Maximum Entropy sampling with MALA and DumbFeature on DCGAN from ```torch_mimicry``` trained on CIFAR10 for 100k iterations.

```bash
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/exp_configs/mmc-dcgan-dumb.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml configs/feature_configs/dumb.yml configs/common.yml --suffix mala
```

Results saved to ```log/inception_feature_DiscriminatorTarget/mmc_dcgan_mala```.

<!-- ![mmc_dcgan_dumb_mala](../log/dumb_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_fid.png) -->



Maximum Entropy sampling with MALA and InceptionFeature on DCGAN from ```torch_mimicry``` trained on CIFAR10 for 100k iterations.

```bash
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/exp_configs/mmc-dcgan-inception.yml configs/targets/discriminator.yml configs/gan_configs/cifar-10-mmc-dcgan.yml configs/feature_configs/inception.yml configs/common.yml --suffix mala
```

Results saved to ```log/inception_feature_DiscriminatorTarget/mmc_dcgan_mala```.

<!-- ![mmc_dcgan_inception_mala](../log/inception_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_fid.png) -->


## 02/07/22

Maximum Entropy sampling with MALA on WGAN-GP-IN and DCGAN trained on CIFAR10.

```bash
TF_CPP_MIN_LOG_LEVEL=3 ./scripts/cifar/inception.sh
```
wgan-gp-in:

<img src="../log/dumb_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_fid.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_fid.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_is.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_is.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_energy.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_energy.png" alt="drawing" width="300"/>

<!-- ![wgan_gp_in_dumb_mala](../log/dumb_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_fid.png)
![wgan_gp_in_inception_mala](../log/inception_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_fid.png)

![wgan_gp_in_dumb_mala](../log/dumb_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_is.png)
![wgan_gp_in_inception_mala](../log/inception_feature_DiscriminatorTarget/wgan_gp_in_mala/figs/wgan_gp_in_mala_is.png) -->


mmc_dcgan:

<img src="../log/dumb_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_fid.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_fid.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_is.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_is.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_energy.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_energy.png" alt="drawing" width="300"/>

<!-- ![dcgan_dumb_mala](../log/dumb_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_fid.png)
![dcgan_inception_mala](../log/inception_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_fid.png)

![dcgan_dumb_mala](../log/dumb_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_is.png)
![dcgan_inception_mala](../log/inception_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_is.png) -->


dcgan:

<img src="../log/dumb_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_fid.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_fid.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_is.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_is.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_energy.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_energy.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_dgz.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/dcgan_mala/figs/dcgan_mala_dgz.png" alt="drawing" width="300"/>

<!-- ![mmc_dcgan_dumb_mala](../log/dumb_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_fid.png)
![mmc_dcgan_inception_mala](../log/inception_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_fid.png)

![mmc_dcgan_dumb_mala](../log/dumb_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_is.png)
![mmc_dcgan_inception_mala](../log/inception_feature_DiscriminatorTarget/mmc_dcgan_mala/figs/mmc_dcgan_mala_is.png) -->

## 06/07/22
dcgan, ula:

<img src="../log/dumb_feature_DiscriminatorTarget/dcgan_ula/figs/dcgan_ula_fid.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/dcgan_ula/figs/dcgan_ula_fid.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/dcgan_ula/figs/dcgan_ula_is.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/dcgan_ula/figs/dcgan_ula_is.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/dcgan_ula/figs/dcgan_ula_energy.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/dcgan_ula/figs/dcgan_ula_energy.png" alt="drawing" width="300"/>

<img src="../log/dumb_feature_DiscriminatorTarget/dcgan_ula/figs/dcgan_ula_dgz.png" alt="drawing" width="300"/>
<img src="../log/inception_feature_DiscriminatorTarget/dcgan_ula/figs/dcgan_ula_dgz.png" alt="drawing" width="300"/>