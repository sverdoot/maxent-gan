import torch
import torch.nn as nn

from soul_gan.models.base import (BaseDiscriminator, BaseGenerator,
                                  ModelRegistry)

                                
# import random

# import numpy as np
# import torch


# def weights_init_1(m):
#     classname = m.__class__.__name__
#     if classname.find("Linear") != -1:
#         m.weight.data.normal_(0.0, 0.02)
#         m.bias.data.fill_(0)
#     elif classname.find("BatchNorm") != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


# def weights_init_2(m):
#     classname = m.__class__.__name__
#     if classname.find("Linear") != -1:
#         std_init = 0.8 * (2 / m.in_features) ** 0.5
#         m.weight.data.normal_(0.0, std=std_init)
#         m.bias.data.fill_(0)


@ModelRegistry.register()
class MLPGenerator(BaseGenerator):
    def __init__(
        self,
        mean,
        std,
        z_dim=2,
        n_layers=4,
        n_hid=100,
        n_out=2,
        non_linear=nn.ReLU(),
    ):
        #super().__init__(mean, std)
        nn.Module.__init__(self)
        self.non_linear = non_linear
        self.n_hid = n_hid
        self.z_dim = z_dim
        self.n_out = n_out
        self.n_layers = n_layers
        layers = [nn.Linear(self.z_dim, self.n_hid), non_linear]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hid, n_hid), non_linear])
        layers.append(nn.Linear(n_hid, n_out))
        self.layers = nn.Sequential(*layers)
        # for i in range(4):
        #    std_init = 0.8 * (2/self.layers[i].in_features)**0.5
        #    torch.nn.init.normal_(self.layers[i].weight, std = std_init)

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        self.inverse_transform = lambda x: (x * std[None, :].to(x.device) + mean[None, :].to(x.device))

    def forward(self, z):
        z = self.layers.forward(z)
        return z

    # def init_weights(self, init_fun=weights_init_1, random_seed=None):
    #     if random_seed is not None:
    #         torch.manual_seed(random_seed)
    #         np.random.seed(random_seed)
    #         random.seed(random_seed)
    #     self.apply(init_fun)


@ModelRegistry.register()
class MLPDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        mean,
        std,
        n_in=2,
        n_layers=4,
        n_hid=100,
        non_linear=nn.ReLU(),
    ):
        nn.Module.__init__(self)
        #super().__init__(mean, std, 'sigmoid')
        self.output_layer = nn.Sigmoid()

        self.non_linear = non_linear
        self.n_hid = n_hid
        self.n_in = n_in
        layers = [nn.Linear(self.n_in, self.n_hid), non_linear]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hid, n_hid), non_linear])
        layers.append(nn.Linear(n_hid, 1))
        self.layers = nn.Sequential(*layers)

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        self.transform = lambda x: (x - mean[None, :].to(x.device)) / std[None, :].to(x.device)

    def forward(self, z):
        z = self.layers.forward(z)
        return z

    # def init_weights(self, init_fun=weights_init_1, random_seed=None):
    #     if random_seed is not None:
    #         torch.manual_seed(random_seed)
    #         np.random.seed(random_seed)
    #         random.seed(random_seed)
    #     self.apply(init_fun)
