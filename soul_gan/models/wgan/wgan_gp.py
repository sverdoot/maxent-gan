import torch.nn as nn

from soul_gan.models.base import BaseDiscriminator, ModelRegistry


@ModelRegistry.register()
class WGANDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        bw=4,
        ch=512,
        output_dim=1,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        super().__init__(mean, std, output_layer="identity")
        self.c0 = nn.Conv2d(3, ch // 8, 3, 1, 1)
        self.c1 = nn.Conv2d(ch // 8, ch // 4, 4, 2, 1)
        self.c1_0 = nn.Conv2d(ch // 4, ch // 4, 3, 1, 1)
        self.c2 = nn.Conv2d(ch // 4, ch // 2, 4, 2, 1)
        self.c2_0 = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.c3 = nn.Conv2d(ch // 2, ch // 1, 4, 2, 1)
        self.c3_0 = nn.Conv2d(ch // 1, ch // 1, 3, 1, 1)

        self.l4 = nn.Linear(bw * bw * ch, output_dim)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        h = self.lrelu(self.c0(x))
        h = self.lrelu(self.c1(h))
        h = self.lrelu(self.c1_0(h))
        h = self.lrelu(self.c2(h))
        h = self.lrelu(self.c2_0(h))
        h = self.lrelu(self.c3(h))
        h = self.lrelu1(self.c3_0(h))
        h = h.view(x.size(0), -1)
        return self.l4(h)
