from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal as MNormal
from torch.distributions import Normal
from tqdm import trange

from .distribution import Distribution, torchDist


class MCMCRegistry:
    registry: Dict = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        def inner_wrapper(function: Callable) -> Callable:
            if name is None:
                name_ = function.__name__
            else:
                name_ = name
            cls.registry[name_] = function
            return function

        return inner_wrapper

    @classmethod
    def __call__(cls, name: str, *args, **kwargs) -> Tuple[torch.FloatTensor, Dict]:
        exec = cls.registry[name]
        kwargs["proposal"] = kwargs.get("proposal", Normal(0, 1))
        kwargs["project"] = kwargs.get("project", lambda _: _)
        chains, meta = exec(*args, **kwargs)
        return chains, meta


@MCMCRegistry.register()
def ula(
    start: torch.FloatTensor,
    target: Union[Distribution, torchDist],
    proposal: Union[Distribution, torchDist],
    n_samples: int,
    burn_in: int,
    project: Callable,
    *,
    step_size: float,
    verbose: bool = False,
    meta=None,
) -> Tuple[torch.FloatTensor, Dict]:
    """
    Unadjusted Langevin Algorithm

    Args:
        start - strating points of shape [n_chains, dim]
        target - target distribution instance with method "log_prob"
        n_samples - number of last samples from each chain to return
        burn_in - number of first samples from each chain to throw away
        step_size - step size for drift term
        verbose - whether to show iterations' bar

    Returns:
        tensor of chains with shape [n_samples, n_chains, dim]
    """
    chains = []

    point = start.clone()
    point.requires_grad_(True)
    point.grad = None

    range_ = trange if verbose else range
    for step_id in range_(n_samples + burn_in):
        point = point.detach().requires_grad_()
        logp = target.log_prob(point)
        grad = torch.autograd.grad(logp.sum(), point)[0]
        noise = torch.randn_like(point, dtype=torch.float).to(point.device)
        noise_scale = (2.0 * step_size) ** 0.5
        point = point + step_size * grad + noise_scale * noise
        point = project(point)
        # point = point.data
        # point.requires_grad_(True)

        if step_id >= burn_in:
            chains.append(point.data)
    chains = torch.stack(chains, 0)
    return chains, meta


@MCMCRegistry.register()
def isir_step(
    start: torch.FloatTensor,
    target,
    proposal: Union[Distribution, torchDist],
    *,
    n_particles: int,
    logp_x=None,
    logq_x=None,
) -> Tuple:
    point = start.clone()
    logp_x = target.log_prob(point) if logp_x is None else logp_x
    logq_x = proposal.log_prob(point) if logq_x is None else logq_x

    particles = proposal.sample((point.shape[0], n_particles - 1))
    log_qs = torch.cat([logq_x[:, None], proposal.log_prob(particles)], 1)
    log_ps = torch.cat([logp_x[:, None], target.log_prob(particles)], 1)
    particles = torch.cat([point[:, None, :], particles], 1)

    log_weights = log_ps - log_qs
    indices = Categorical(logits=log_weights).sample()

    x = particles[np.arange(point.shape[0]), indices]

    return x, particles.detach(), log_ps, log_qs, indices


@MCMCRegistry.register()
def isir(
    start: torch.FloatTensor,
    target,
    proposal: Union[Distribution, torchDist],
    n_samples: int,
    burn_in: int,
    project: Callable,
    *,
    n_particles: int,
    verbose: bool = False,
    meta=None,
) -> Tuple[torch.FloatTensor, Dict]:
    """
    Iterated Sampling Importance Resampling

    Args:
        start - strating points of shape [n_chains x dim]
        target - target distribution instance with method "log_prob"
        proposal - proposal distribution instance with methods "log_prob" and "sample"
        n_samples - number of last samples from each chain to return
        burn_in - number of first samples from each chain to throw away
        n_particles - number of particles including one from previous step
        verbose - whether to show iterations' bar

    Returns:
        tensor of chains with shape [n_samples, n_chains, dim], meta
    """
    chains = []

    if meta:
        meta["sir_accept"] = meta.get("sir_accept", [])
    else:
        meta = dict(sir_accept=[])

    point = start.clone()
    point = project(start)
    logp_x = target.log_prob(point)
    logq_x = proposal.log_prob(point)

    range_ = trange if verbose else range
    for step_id in range_(n_samples + burn_in):
        point, log_qs, log_ps, indices = isir_step(
            point,
            target,
            proposal,
            n_particles=n_particles,
            logp_x=logp_x,
            logq_x=logq_x,
        )
        logp_x = log_ps[np.arange(point.shape[0]), indices]
        logq_x = log_qs[np.arange(point.shape[0]), indices]
        meta["sir_accept"].append((indices != 0).float().mean().item())
        if step_id >= burn_in:
            chains.append(point.detach().data.cpu().clone())
    chains = torch.stack(chains, 0)
    return chains, meta


def heuristics_step_size(
    mean_acceptance: float,
    target_acceptance: float,
    step_size: float,
    factor: float = 1.05,
    tol: float = 0.03,
):
    if mean_acceptance - target_acceptance > tol:
        return step_size * factor
    if target_acceptance - mean_acceptance > tol:
        return step_size / factor
    return step_size


@MCMCRegistry.register()
def mala(
    start: torch.FloatTensor,
    target: Union[Distribution, torchDist],
    proposal: Union[Distribution, torchDist],
    n_samples: int,
    burn_in: int,
    project: Callable,
    *,
    step_size: float,
    verbose: bool = False,
    target_acceptance=None,
    meta=[],
) -> Tuple[torch.FloatTensor, Dict]:
    """
    Metropolis-Adjusted Langevin Algorithm with Normal proposal

    Args:
        start - strating points of shape [n_chains x dim]
        target - target distribution instance with method "log_prob"
        step_size - step size for drift term
        verbose - whether show iterations' bar

    Returns:
        sequence of slices per each iteration, meta
    """
    std_normal = MNormal(
        torch.zeros(start.shape[-1], device=start.device),
        torch.eye(start.shape[-1], device=start.device),
    )
    chains = []
    if meta:
        meta["mh_accept"] = meta.get("mh_accept", [])
        meta["step_size"] = meta.get("step_size", [])
    else:
        meta = dict(mh_accept=[], step_size=[])

    x = start.clone()
    x.requires_grad_(True)
    x.grad = None

    x = x.detach().requires_grad_()
    logp_x = target.log_prob(x)
    grad_x = torch.autograd.grad(logp_x.sum(), x)[0]

    range_ = trange if verbose else range
    for step_id in range_(n_samples + burn_in):
        noise = torch.randn_like(x)
        y = x + step_size * grad_x + noise * (2 * step_size) ** 0.5
        y = project(y)

        y = y.detach().requires_grad()
        logp_y = target.log_prob(y)
        grad_y = torch.autograd.grad(logp_y.sum(), y)[0]

        log_qyx = std_normal.log_prob(noise)
        log_qxy = std_normal.log_prob(
            (x - y - step_size * grad_y) / (2 * step_size) ** 0.5
        )

        accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
        mask = torch.rand_like(accept_prob) < accept_prob

        with torch.no_grad():
            x[mask, :] = y[mask, :]
            logp_x[mask] = logp_y[mask]
            grad_x[mask] = grad_y[mask]
        # x = x.data
        # x.requires_grad_()

        meta["mh_accept"].append(mask.float().mean().item())
        if target_acceptance:
            step_size = heuristics_step_size(
                np.mean(meta["mh_accept"]), target_acceptance, step_size
            )
        meta["step_size"].append(step_size)
        if step_id >= burn_in:
            chains.append(x.detach().data.cpu().clone())
    chains = torch.stack(chains, 0)
    return chains, meta


@MCMCRegistry.register()
def ex2mcmc(
    start: torch.FloatTensor,
    target: Union[Distribution, torchDist],
    proposal: Union[Distribution, torchDist],
    n_samples: int,
    burn_in: int,
    project: Callable,
    *,
    step_size: float,
    n_particles: int,
    target_acceptance_rate: float = False,
    verbose: bool = False,
    meta=None,
):
    chains = []
    if meta:
        meta["sir_accept"] = meta.get("sir_accept", [])
        meta["mh_accept"] = meta.get("mh_accept", [])
        meta["step_size"] = meta.get("step_size", [])
    else:
        meta = dict(sir_accept=[], mh_accept=[], step_size=[])

    x = start.clone()
    x.requires_grad_(True)
    x.grad = None

    range_ = trange if verbose else range
    for _ in range_(n_samples + burn_in):
        xs, meta = isir(
            x, target, proposal, 1, 0, project, n_particles=n_particles, meta=meta
        )
        xs, meta = mala(
            xs[1],
            target,
            proposal,
            1,
            0,
            project,
            step_size=step_size,
            target_acceptance=target_acceptance_rate,
            meta=meta,
        )
        x = xs[1].data
        x.requires_grad_()
        chains.append(x.clone().detach())

    chains = torch.stack(chains, 0)
    return chains, meta


# class FlowMCMC:
#     def __init__(self, target, proposal, flow, mcmc_call: Callable, **kwargs):
#         self.flow = flow
#         self.proposal = proposal
#         self.target = target
#         self.batch_size = kwargs.get("batch_size", 64)
#         self.mcmc_call = mcmc_call
#         self.grad_clip = kwargs.get("grad_clip", 1.0)
#         self.jump_tol = kwargs.get("jump_tol", 1e2)
#         optimizer = kwargs.get("optimizer", "adam")
#         loss = kwargs.get("loss", "mix_kl")

#         if isinstance(loss, (Callable, torch.nn.Module)):
#             self.loss = loss
#         elif isinstance(loss, str):
#             self.loss = get_loss(loss)(self.target, self.proposal, self.flow)
#         else:
#             ValueError

#         lr = kwargs.get("lr", 1e-3)
#         wd = kwargs.get("wd", 1e-4)
#         if isinstance(optimizer, torch.optim.Optimizer):
#             self.optimizer = optimizer
#         elif isinstance(optimizer, str):
#             if optimizer.lower() == "adam":
#                 self.optimizer = torch.optim.Adam(
#                     flow.parameters(), lr=lr, weight_decay=wd
#                 )

#         self.loss_hist = []

#     def train_step(self, inp=None, alpha=0.5, do_step=True, inv=True):
#         if do_step:
#             self.optimizer.zero_grad()
#         if inp is None:
#             inp = self.proposal.sample((self.batch_size,))
#         elif inv:
#             inp, _ = self.flow.inverse(inp)

#         out = self.mcmc_call(inp, self.target, self.proposal, flow=self.flow)
#         if isinstance(out, Tuple):
#             acc_rate = out[1].mean()
#             out = out[0]
#         else:
#             acc_rate = 1
#         out = out[-1]

#         nll = -self.target(out).mean().item()

#         if do_step:
#             loss_est, loss = self.loss(out, acc_rate=acc_rate, alpha=alpha)

#             if (
#                 len(self.loss_hist) > 0
#                 and loss.item() - self.loss_hist[-1] > self.jump_tol
#             ) or torch.isnan(loss):
#                 print("KL wants to jump, terminating learning")
#                 return out, nll

#             self.loss_hist = self.loss_hist[-500:] + [loss_est.item()]
#             self.optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 self.flow.parameters(),
#                 self.grad_clip,
#             )
#             self.optimizer.step()

#         return out, nll

#     def train(self, n_steps=100, start_optim=10, init_points=None, alpha=None):
#         samples = []
#         inp = self.proposal.sample((self.batch_size,))

#         neg_log_likelihood = []

#         for step_id in trange(n_steps):
#             if alpha is not None:
#                 if isinstance(alpha, Callable):
#                     a = alpha(step_id)
#                 elif isinstance(alpha, float):
#                     a = alpha
#             else:
#                 a = min(1.0, 3 * step_id / n_steps)
#             out, nll = self.train_step(
#                 alpha=a,
#                 do_step=step_id >= start_optim,
#                 inp=init_points
#                 if step_id == 0 and init_points is not None
#                 else inp,
#                 inv=True,
#             )
#             inp = out.detach().requires_grad_()
#             samples.append(inp.detach().cpu())

#             neg_log_likelihood.append(nll)

#         return samples, neg_log_likelihood

#     def sample(self):
#         pass
