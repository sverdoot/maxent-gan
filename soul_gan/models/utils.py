from pathlib import Path
from typing import Tuple

import torch
from torchvision import transforms

from soul_gan.models import ModelRegistry
from soul_gan.utils.general_utils import ROOT_DIR, DotConfig


def load_gan(
    config: DotConfig, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    gen = ModelRegistry.create_model(
        config.generator.name, **config.generator.params
    ).to(device)
    state_dict = torch.load(
        Path(ROOT_DIR, config.generator.ckpt_path, map_location=device)
    )
    gen.load_state_dict(state_dict)

    dis = ModelRegistry.create_model(
        config.discriminator.name, **config.discriminator.params
    ).to(device)
    state_dict = torch.load(
        Path(ROOT_DIR, config.discriminator.ckpt_path, map_location=device)
    )
    dis.load_state_dict(state_dict)

    gen.eval()
    dis.eval()

    return gen, dis


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
