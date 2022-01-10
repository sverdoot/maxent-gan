from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution as torchDist
from tqdm import trange


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
        dgz = dis(gen(z)).squeeze()
        logp_z = proposal.log_prob(z)
        # print(logp_z.mean().item(), dgz.mean().item())
        assert dgz.shape == logp_z.shape
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


def estimate_log_norm_constant(
    gen: nn.Module,
    dis: nn.Module,
    n_pts: int,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> float:
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

        dgz = dis(gen(z)).squeeze()
        norm_const += (dgz).exp().sum().item()
    norm_const /= n_pts

    log_norm_const = float(np.log(norm_const))
    return log_norm_const


def harmonic_mean_estimate(
    dis: nn.Module,
    x: Union[np.ndarray, torch.FloatTensor],
    label: Union[np.ndarray, torch.FloatTensor],
    device,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> float:
    batch_size = batch_size if batch_size else len(x)
    inv_norm_const = 0
    # if verbose:
    #     bar = tqdm
    # else:
    #     bar = range
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    for i, x_batch in enumerate(torch.split(x, batch_size)):
        label_batch = label[i * batch_size : (i + 1) * batch_size]
        dis.label = label_batch.to(device)

        dgz = dis(x_batch.to(device)).squeeze()
        inv_norm_const += (-dgz).exp().sum().item()
    inv_norm_const /= len(x)

    log_norm_const = float(-np.log(inv_norm_const))
    return log_norm_const
