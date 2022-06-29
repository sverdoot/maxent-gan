from typing import Tuple

from torch_mimicry.nets import sngan

from maxent_gan.models.base import BaseDiscriminator, BaseGenerator, ModelRegistry


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

    @property
    def penult_layer(self):
        return self.dis.activation

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.dis.load_state_dict(state_dict["model_state_dict"], strict=strict)

    def forward(self, x):
        return self.dis(x)
