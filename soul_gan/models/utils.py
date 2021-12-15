from pathlib import Path
from typing import Tuple

import torch
from torchvision import transforms
from tqdm import trange

from soul_gan.models import ModelRegistry
from soul_gan.utils.general_utils import ROOT_DIR, DotConfig


def stabilize_dis(dis, im_size=32, iters=5000, device=0):
    for _ in trange(iters):
        x = torch.rand(10, 3, im_size, im_size, device=device)
        _ = dis(x)


def stabilize_gen(gen, iters=500):
    for _ in trange(iters):
        x = gen.prior.sample((100,))
        _ = gen(x)


def load_gan(
    config: DotConfig, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    gen = ModelRegistry.create_model(
        config.generator.name, **config.generator.params
    ).to(device)
    state_dict = torch.load(
        Path(ROOT_DIR, config.generator.ckpt_path, map_location=device)
    )
    gen.load_state_dict(state_dict, strict=True)

    dis = ModelRegistry.create_model(
        config.discriminator.name, **config.discriminator.params
    ).to(device)
    state_dict = torch.load(
        Path(ROOT_DIR, config.discriminator.ckpt_path, map_location=device)
    )
    dis.load_state_dict(state_dict, strict=True)

    if config.dp:
        gen = torch.nn.DataParallel(gen)
        dis = torch.nn.DataParallel(dis)
        dis.transform = dis.module.transform
        gen.inverse_transform = gen.module.inverse_transform
        gen.z_dim = gen.module.z_dim

    if config.prior == "normal":
        prior = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(gen.z_dim).to(device), torch.eye(gen.z_dim).to(device)
        )
        prior.project = lambda z: z
    elif config.prior == "uniform":
        prior = torch.distributions.uniform.Uniform(
            -torch.ones(gen.z_dim).to(device), torch.ones(gen.z_dim).to(device)
        )
        prior.project = lambda z: torch.clip(z, -1 + 1e-9, 1 - 1e-9)
        prior.log_prob = lambda z: torch.zeros_like(z)
    else:
        raise KeyError
    gen.prior = prior

    # if True: #config.discriminator.thermalize:
    stabilize_dis(dis, device=device)
    stabilize_gen(gen)

    # gen.eval()
    # dis.eval()

    return gen, dis


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-10)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
