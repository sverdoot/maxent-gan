# CHANGELOG

## 29/06/22

* Added CHANGELOG.md
* Added checkpoint for DCGAN with non-saturating Vanilla loss from torch_mimicry trained for 100k iterations on CIFAR10
* Added InceptionFeature (```maxent_gan.feature.InceptionFeature```)
* Updated code for MCMC methods (```maxent_gan.mcmc```)
* Updated code for MaxEntSampler (```maxent_gan.sample.MaxEntSampler```)
* Updated code for distributions (```maxent_gan.distribution```)
* Added ```docs/revival_log.md``` for logging experiments

## 02/07/22

* Added changes to MCMC (```maxent_gan.mcmc```): made more computationally efficient and added masks to returns
* Fixed feature output averaging for MCMC methods different from ULA (```maxent_gan.feature.feature.BaseFeature```) for propoer weight updates
* Corresponding changes to MaxEntSampler (```maxent_gan.sample.MaxEntSampler```))
* Added WGAN-GP (with Instance Normalization) (```checkpoints/wgan_gp_in```))
* Updated ```docs/revival_log.md```

## 04/07/22

* More efficient target log prob computation: eliminate repeats of ```model.forward``` for same inputs (```maxent_gan.models.base.MemoryModel```)
* Fixed bug with seed in ```run.py```

## 07/07/22

* Added training with "meta-objective" (backprop through sampling in the latent space) (```train_meta.py```, ```maxent_gan.utils.train.trainer_meta```) 