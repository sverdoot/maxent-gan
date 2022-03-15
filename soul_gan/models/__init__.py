# flake8: noqa
from .base import ModelRegistry  # noqa: F401
from .dcgan import DCGANDiscriminator, DCGANGenerator  # noqa: F401
from .dcgan import PresDCGANDiscriminator, PresDCGANGenerator  # noqa: F401
from .mimicry import MMCSNDiscriminator, MMCSNGenerator
from .mlp import MLPDiscriminator, MLPGenerator  # noqa: F401
from .sngan import SN_DCGAN_Generator  # noqa: F401
from .sngan import SN_DCGAN_Discriminator, SN_ResNet_Generator32
from .wgan import WGANDiscriminator  # noqa: F401
