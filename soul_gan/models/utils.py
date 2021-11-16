from pathlib import Path
from typing import Tuple

import torch

from soul_gan.models import ModelRegistry
from soul_gan.utils.general_utils import DotConfig


def load_gan(
    config: DotConfig, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    gen = ModelRegistry.create_model(
        config.generator.name, **config.generator.params
    ).to(device)
    state_dict = torch.load(
        Path(config.generator.ckpt_path, map_location=device)
    )
    gen.load_state_dict(state_dict)

    dis = ModelRegistry.create_model(
        config.discriminator.name, **config.discriminator.params
    ).to(device)
    state_dict = torch.load(
        Path(config.discriminator.ckpt_path, map_location=device)
    )
    dis.load_state_dict(state_dict)

    gen.eval()
    dis.eval()

    return gen, dis
