import random
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
# import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
import torchvision
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
import numpy as np
from torch import autograd
from maxent_gan.models.base import BaseDiscriminator, BaseGenerator, ModelRegistry
torch.manual_seed(430156)


D_SCALE = 1.3
ImageSize=32
batchSize=32

# elif Dset == 'cifar10':
    # dataset = dset.CIFAR10(root=DataRoot, download=True,
    #                        transform=transforms.Compose([
    #                            transforms.Resize(ImageSize),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                        ]))
#     # nc=3

# assert dataset
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
#                                          shuffle=True, num_workers=2)


channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_in', nonlinearity='relu')

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, shape, stride=1, bn=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_out', nonlinearity='relu')
        norm = lambda c : (nn.Sequential() if not bn else nn.BatchNorm2d(c))
        if stride == 1:
            self.model = nn.Sequential(
                norm(in_channels),
                nn.ReLU(),
                self.conv1,
                norm(out_channels),
                nn.ReLU(),
                self.conv2,
                )
        else:
            self.model = nn.Sequential(
                norm(in_channels),
                nn.ReLU(),
                self.conv1,
                norm(out_channels),
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, nn.init.calculate_gain('linear'))

            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform(self.bypass_conv.weight.data, nn.init.calculate_gain('linear'))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=256
DISC_SIZE=int(256*D_SCALE)


@ModelRegistry.register()
class PPOGenerator(BaseGenerator):
    def __init__(self, z_dim=128, mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5), ):
        super().__init__(mean, std)
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform(self.final.weight.data, nn.init.calculate_gain('tanh'))

        self.model = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))

    # def load_state_dict(self, state_dict, strict: bool = True):
    #     out = self.load_state_dict(state_dict, strict=strict)

    #     return out

@ModelRegistry.register()
class PPODiscriminator(BaseDiscriminator):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        output_layer="identity",):
        
        super().__init__(mean, std, output_layer)
        # self.model = nn.Sequential(
        self.r1 = FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2)
        self.r2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 16, stride=2)
        self.r3 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 8)
        self.r4 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 8)
            # nn.ReLU(),
        self.pool = nn.AvgPool2d(8)
            # )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, nn.init.calculate_gain('linear'))

    def forward(self, x, d1=0.2, d2=0.5, d3=0.5):
        x = F.dropout(self.r2(self.r1(x)), p=d1)
        x = F.dropout(self.r3(x), p=d2)
        o2 = self.pool(F.relu(F.dropout(self.r4(x), p=d3), inplace=True))
        
        return self.fc(o2.view(-1,DISC_SIZE))
        # o2.squeeze()

    # def load_state_dict(self, state_dict, strict: bool = True):
    #     out = self.load_state_dict(state_dict, strict=strict)

    #     return out

class PPODiscriminator_D(nn.Module):
    def __init__(self):
        super(PPODiscriminator_D, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 16, stride=2, bn = True),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 8, bn = True),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 8, bn = True),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, nn.init.calculate_gain('linear'))

    def forward(self, x):
        return F.sigmoid(self.fc(self.model(x).view(-1,DISC_SIZE)))
