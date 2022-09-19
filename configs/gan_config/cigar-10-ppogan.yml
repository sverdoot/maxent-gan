gan_config:
  dp: true
  dataset: 
    name: cifar10
    params: {}
  train_transform:
    Normalize:
      mean: &mean [0.5, 0.5, 0.5]
      std: &std [0.5, 0.5, 0.5]
  prior: normal
  eval: true
  generator:
    name: PPOGenerator
    params:
      mean: *mean
      std: *std
    ckpt_path: checkpoints/ppogan_chkpt/G_iter60000.pt
  discriminator:
    name: PPODiscriminator
    params:
      mean: *mean
      std: *std
      output_layer: identity
    ckpt_path: checkpoints/ppogan_chkpt/D_iter60000.pt
