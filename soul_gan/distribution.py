from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union
from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution as torchDist


class Distribution(ABC):
    """Abstract class for distribution"""

    @abstractmethod
    def __call__(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Computes log probability of input z"""
        raise NotImplementedError


class GANTarget(Distribution):
    def __init__(
        self,
        gen: nn.Module,
        dis: nn.Module,
        # proposal: Union[Distribution, torchDist],
    ):
        self.gen = gen
        self.dis = dis
        self.proposal = gen.prior

    @staticmethod
    def latent_target(
        z: torch.FloatTensor,
        gen: nn.Module,
        dis: nn.Module,
        proposal: Union[Distribution, torchDist],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        dgz = dis(gen(z))
        logp_z = proposal.log_prob(z)
        # print(logp_z.mean(), dgz.mean())
        log_prob = (logp_z + dgz) / 1.0

        return log_prob, logp_z, dgz

    def __call__(self, z: torch.FloatTensor) -> torch.FloatTensor:
        logp = self.latent_target(z, self.gen, self.dis, self.proposal)[0]
        return logp

    def project(self, z):
        return self.proposal.project(z)


def grad_log_prob(
    point: torch.FloatTensor,
    log_dens: Union[Distribution, Callable],
    # x: Optional[Any] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    point = point.detach().requires_grad_()
    # if x is not None:
    #     log_prob = log_dens(z=point, x=x)
    # else:
    log_prob = log_dens(point)
    grad = torch.autograd.grad(log_prob.sum(), point)[0]
    return log_prob, grad


def estimate_log_norm_constant(gen: nn.Module, dis: nn.Module, n_pts: int, batch_size: Optional[int]=None, verbose: bool = False) -> float:
    batch_size = batch_size if batch_size else n_pts
    norm_const = 0
    if verbose:
        bar = trange
    else:
        bar = range

    for _ in bar(0, n_pts, batch_size):
        z = gen.prior.sample((batch_size,))
        label = gen.sample_label(batch_size, z.device)
        gen.label = label
        dis.label = label

        dgz = dis(gen(z))
        norm_const += (dgz).exp().sum().item()
    norm_const /= n_pts

    log_norm_const = float(np.log(norm_const))
    return log_norm_const


# def estimate_log_norm_constant_2(gen, dis, z, label, device, batch_size: Optional[int]=None, verbose: bool = False) -> float:
#     batch_size = batch_size if batch_size else len(z)
#     inv_norm_const = 0
#     if verbose:
#         bar = trange
#     else:
#         bar = range

#     for i, z_batch in enumerate(np.split(z, batch_size)):
#         label_batch = label[i*batch_size: (i+1)*batch_size]
#         gen.label = label_batch
#         dis.label = label_batch

#         dgz = dis(gen(torch.from_numpy(z_batch).to(device)))
#         inv_norm_const += (gen.prior.log_prob(z_batch)+dgz).exp().sum().item()
#     inv_norm_const /= len(z)

#     log_norm_const = float(np.log(norm_const))
#     return log_norm_const