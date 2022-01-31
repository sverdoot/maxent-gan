import os
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch_mimicry as mmc
from torch_mimicry.nets import sngan

from soul_gan.models.base import (BaseDiscriminator, BaseGenerator,
                                  ModelRegistry)


@ModelRegistry.register()
class MMCSNGenerator(BaseGenerator):
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ):
        super().__init__(mean, std)
        self.gen = sngan.SNGANGenerator64()
        self.z_dim = self.gen.nz

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.gen.load_state_dict(state_dict["model_state_dict"], strict=strict)

    def forward(self, z):
        return self.gen(z)


@ModelRegistry.register()
class MMCSNDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        output_layer="identity",
    ):
        super().__init__(mean, std, output_layer)
        self.dis = sngan.SNGANDiscriminator64()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.dis.load_state_dict(state_dict["model_state_dict"], strict=strict)

    def forward(self, x):
        return self.dis(x)
