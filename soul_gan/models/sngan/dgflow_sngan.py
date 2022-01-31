import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import spectral_norm

from soul_gan.models.base import (BaseDiscriminator, BaseGenerator,
                                  ModelRegistry)


def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode="nearest")


def upsample_conv(x, conv):
    return conv(_upsample(x))


@ModelRegistry.register()
class SN_DCGAN_Generator(BaseGenerator):
    def __init__(
        self,
        n_hidden=128,
        bw=4,
        ch=512,
        nz=128,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        super().__init__(mean, std)
        self.z_dim = nz
        self.ch = ch
        self.bw = bw
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.l0 = nn.Linear(n_hidden, bw * bw * ch)
        self.dc1 = nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1)
        self.dc2 = nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1)
        self.dc3 = nn.ConvTranspose2d(ch // 4, ch // 8, 4, 2, 1)
        self.dc4 = nn.ConvTranspose2d(ch // 8, 3, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(bw * bw * ch, eps=2e-5, momentum=0.1)
        self.bn1 = nn.BatchNorm2d(ch // 2, eps=2e-5, momentum=0.1)
        self.bn2 = nn.BatchNorm2d(ch // 4, eps=2e-5, momentum=0.1)
        self.bn3 = nn.BatchNorm2d(ch // 8, eps=2e-5, momentum=0.1)

    def forward(self, z, **kwargs):
        h = self.l0(z)
        h = h.view(-1, self.ch * self.bw * self.bw, 1, 1)
        h = self.relu(self.bn0(h))
        h = h.view(-1, self.ch, self.bw, self.bw)
        h = self.relu(self.bn1(self.dc1(h)))
        h = self.relu(self.bn2(self.dc2(h)))
        h = self.relu(self.bn3(self.dc3(h)))
        o = self.tanh(self.dc4(h))
        return o


@ModelRegistry.register()
class SN_DCGAN_Discriminator(BaseDiscriminator):
    def __init__(
        self,
        bw=4,
        ch=512,
        output_dim=1,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        output_layer="identity",
    ):
        super().__init__(mean, std, output_layer)
        c0_0 = nn.Conv2d(3, ch // 8, 3, 1, 1)
        self.c0_0 = spectral_norm(c0_0)

        c0_1 = nn.Conv2d(ch // 8, ch // 4, 4, 2, 1)
        self.c0_1 = spectral_norm(c0_1)

        c1_0 = nn.Conv2d(ch // 4, ch // 4, 3, 1, 1)
        self.c1_0 = spectral_norm(c1_0)

        c1_1 = nn.Conv2d(ch // 4, ch // 2, 4, 2, 1)
        self.c1_1 = spectral_norm(c1_1)

        c2_0 = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.c2_0 = spectral_norm(c2_0)

        c2_1 = nn.Conv2d(ch // 2, ch // 1, 4, 2, 1)
        self.c2_1 = spectral_norm(c2_1)

        c3_0 = nn.Conv2d(ch // 1, ch // 1, 3, 1, 1)
        self.c3_0 = spectral_norm(c3_0)

        l4 = nn.Linear(bw * bw * ch, output_dim)
        self.l4 = spectral_norm(l4)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        h = self.lrelu(self.c0_0(x))
        h = self.lrelu(self.c0_1(h))
        h = self.lrelu(self.c1_0(h))
        h = self.lrelu(self.c1_1(h))
        h = self.lrelu(self.c2_0(h))
        h = self.lrelu(self.c2_1(h))
        h = self.lrelu(self.c3_0(h))
        h = h.view(x.size(0), -1)
        return self.l4(h)


@ModelRegistry.register()
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        ksize=3,
        pad=1,
        activation=nn.ReLU(),
        upsample=False,
        n_classes=0,
    ):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        self.c1 = c1
        c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        self.c2 = c2
        # if n_classes > 0:
        #     b1 = CategoricalConditionalBatchNorm2d(
        #         n_classes, in_channels, eps=2e-5, momentum=0.1)
        #     copy_CategoricalConditionalBatchNorm2d(b1, B.b1)
        #     self.b1 = b1
        #     b2 = CategoricalConditionalBatchNorm2d(
        #         n_classes, hidden_channels, eps=2e-5, momentum=0.1)
        #     copy_CategoricalConditionalBatchNorm2d(b2, B.b2)
        #     self.b2 = b2
        # else:
        b1 = nn.BatchNorm2d(in_channels, eps=2e-5, momentum=0.1)
        self.b1 = b1
        b2 = nn.BatchNorm2d(hidden_channels, eps=2e-5, momentum=0.1)
        self.b2 = b2
        if self.learnable_sc:
            c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            self.c_sc = c_sc

    def residual(self, x, y=None, **kwargs):
        h = x
        if kwargs.get("resize") is not None:
            del kwargs["resize"]
        h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None, z=None, **kwargs):
        return self.residual(x, y, **kwargs) + self.shortcut(x)


@ModelRegistry.register()
class SN_ResNet_Generator32(BaseGenerator):
    def __init__(
        self,
        ch=256,
        dim_z=128,
        bottom_width=4,
        activation=nn.ReLU(),
        n_classes=0,
        nz=128,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        super().__init__(mean, std)
        self.z_dim = nz
        self.nz = nz
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch)
        self.block2 = ResidualBlock(
            ch, ch, activation=activation, upsample=True, n_classes=n_classes
        )
        self.block3 = ResidualBlock(
            ch, ch, activation=activation, upsample=True, n_classes=n_classes
        )
        self.block4 = ResidualBlock(
            ch, ch, activation=activation, upsample=True, n_classes=n_classes
        )
        self.b5 = nn.BatchNorm2d(ch, eps=2e-5, momentum=0.1)
        self.c5 = nn.Conv2d(ch, 3, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z, y=None, **kwargs):
        h = z
        h = self.l1(h)
        h = h.view((h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.b5(h)
        h = self.activation(h)
        h = self.tanh(self.c5(h))
        return h
