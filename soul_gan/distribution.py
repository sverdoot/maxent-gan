import numpy as np
import torch
from easydict import EasyDict as edict
from torch.distributions import Uniform
from tqdm import tqdm, trange


def grad_energy(point, log_dens, x=None):
    point = point.detach().requires_grad_()
    if x is not None:
        energy = -log_dens(z=point, x=x)
    else:
        energy = -log_dens(point)
    grad = torch.autograd.grad(energy.sum(), point)[0]
    return energy, grad


class GANTarget(object):
    def __init__(self, gen, dis, proposal):
        self.gen = gen
        self.dis = dis
        self.proposal = proposal

    @staticmethod
    def latent_target(z, gen, dis, proposal):
        dgz = dis(gen(z))
        logp_z = proposal(z)
        energy = -(logp_z + dgz)
        return -energy, logp_z, dgz

    def __call__(self, z):
        logp = self.latent_target(
            z, self.gen, self.dis, self.proposal
            )[0]
        return logp
