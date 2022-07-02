from typing import Tuple

import torch
import torch.nn as nn

from maxent_gan.models.base import BaseDiscriminator, BaseGenerator, ModelRegistry


@ModelRegistry.register()
class WGANGeneratorIN(BaseGenerator):
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        z_dim: int = 100,
    ):
        super().__init__(mean, std)
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.z_dim = z_dim
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(
                in_channels=z_dim, out_channels=1024, kernel_size=4, stride=1, padding=0
            ),
            nn.InstanceNorm2d(num_features=1024, affine=True),
            nn.ReLU(True),
            # State (1024x4x4)
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(num_features=512, affine=True),
            nn.ReLU(True),
            # State (512x8x8)
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(num_features=256, affine=True),
            nn.ReLU(True),
            # State (256x16x16)
            nn.ConvTranspose2d(
                in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1
            ),
        )

        self.output = nn.Tanh()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.main_module(x[:, :, None, None])
        return self.output(x)


@ModelRegistry.register()
class WGANDiscriminatorIN(BaseDiscriminator):
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(mean, std, output_layer="identity")
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective
            # (WGAN with gradient penalty) is no longer valid in this setting, since we penalize
            # the norm of the critic's gradient with respect to each input independently and not
            # the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance
            #  normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(
                in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x16)
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (512x8x8)
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid
            # at the output of D.
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)
