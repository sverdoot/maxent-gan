from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pyro.infer import HMC, MCMC
from torch.distributions import Normal  # noqa: F401
from torch.distributions import Categorical
from torch.distributions import Distribution as torchDist
from torch.nn import functional as F
from tqdm import trange

from maxent_gan.distribution import Distribution


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
        # kwargs["proposal"] = kwargs.get("proposal", Normal(0, 1))
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
    meta: Optional[Dict] = None,
    keep_graph: bool = False,
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
    meta = meta or dict()

    point = start.clone()
    point.requires_grad_(True)
    point.grad = None

    pbar = trange if verbose else range
    for step_id in pbar(n_samples + burn_in):
        logp = target.log_prob(point)
        grad = torch.autograd.grad(
            logp.sum(), point, create_graph=keep_graph, retain_graph=keep_graph
        )[0]
        noise = torch.randn_like(point, dtype=torch.float).to(point.device)
        noise_scale = (2.0 * step_size) ** 0.5

        point = point + step_size * grad + noise_scale * noise
        point = project(point)

        if not keep_graph:
            point = point.detach().requires_grad_()
        if step_id >= burn_in:
            chains.append(point)
    chains = torch.stack(chains, 0)
    meta["mask"] = torch.ones(point.shape[0], dtype=torch.bool)

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
    meta: Optional[Dict] = None,
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
    point = start.clone()
    # point = project(start)

    meta = meta or dict()
    meta["sir_accept"] = meta.get("sir_accept", [])
    meta["logp"] = meta.get("logp", target.log_prob(point))
    logp_x = meta["logp"]
    logq_x = proposal.log_prob(point)

    pbar = trange if verbose else range
    for step_id in pbar(n_samples + burn_in):
        point, _, log_qs, log_ps, indices = isir_step(
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
            chains.append(point)
    chains = torch.stack(chains, 0)

    meta["logp"] = logp_x
    meta["mask"] = F.one_hot(indices, num_classes=n_particles).to(bool).detach().cpu()

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
    meta: Optional[Dict] = None,
    keep_graph: bool = False,
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
    if n_samples + burn_in <= 0:
        raise ValueError("Number of steps might be positive")

    chains = []
    point = start.clone()
    point.requires_grad_()
    point.grad = None

    meta = meta or dict()
    meta["mh_accept"] = meta.get("mh_accept", [])
    meta["step_size"] = meta.get("step_size", [])

    if "grad" not in meta:
        logp_x = target.log_prob(point)
        if keep_graph:
            grad_x = torch.autograd.grad(
                logp_x.sum(), point, create_graph=keep_graph, retain_graph=keep_graph
            )[0]
        else:
            grad_x = torch.autograd.grad(logp_x.sum(), point)[0].detach()
        meta["logp"] = logp_x
        meta["grad"] = grad_x
    logp_x = meta["logp"]
    grad_x = meta["grad"]

    pbar = trange if verbose else range
    for step_id in pbar(n_samples + burn_in):
        noise = proposal.sample(point.shape[:-1])
        proposal_point = point + step_size * grad_x + noise * (2 * step_size) ** 0.5
        proposal_point = project(proposal_point)
        if not keep_graph:
            proposal_point = proposal_point.detach().requires_grad_()

        logp_y = target.log_prob(proposal_point)
        grad_y = torch.autograd.grad(
            logp_y.sum(),
            proposal_point,
            create_graph=keep_graph,
            retain_graph=keep_graph,
        )[
            0
        ]  # .detach()

        log_qyx = proposal.log_prob(noise)
        log_qxy = proposal.log_prob(
            (point - proposal_point - step_size * grad_y) / (2 * step_size) ** 0.5
        )

        accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
        mask = torch.rand_like(accept_prob) < accept_prob
        mask = mask.detach()

        if keep_graph:
            mask_f = mask.float()
            point = point * (1 - mask_f)[:, None] + proposal_point * mask_f[:, None]
            logp_x = logp_x * (1 - mask_f) + logp_y * mask_f
            grad_x = grad_x * (1 - mask_f)[:, None] + grad_y * mask_f[:, None]
        else:
            with torch.no_grad():
                point[mask, :] = proposal_point[mask, :]
                logp_x[mask] = logp_y[mask]
                grad_x[mask] = grad_y[mask]

        meta["mh_accept"].append(mask.float().mean().item())
        if target_acceptance:
            step_size = heuristics_step_size(
                meta["mh_accept"][-1], target_acceptance, step_size
            )
        meta["step_size"].append(step_size)

        if not keep_graph:
            point = point.detach().requires_grad_()
        if step_id >= burn_in:
            chains.append(point)
    chains = torch.stack(chains, 0)

    meta["logp"] = logp_x
    meta["grad"] = grad_x
    meta["mask"] = mask.detach().cpu()

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
    n_mala_steps: int = 1,
    target_acceptance: float = False,
    verbose: bool = False,
    meta: Optional[Dict] = None,
    keep_graph: bool = False,
):
    chains = []
    point = start.clone()
    point.requires_grad_(True)
    point.grad = None

    meta = meta or dict()
    meta["sir_accept"] = meta.get("sir_accept", [])
    meta["mh_accept"] = meta.get("mh_accept", [])
    meta["step_size"] = meta.get("step_size", [])

    pbar = trange if verbose else range
    for step_id in pbar(n_samples + burn_in):
        points, meta = isir(
            point, target, proposal, 1, 0, project, n_particles=n_particles, meta=meta
        )
        # meta["grad"] = torch.autograd.grad(meta["logp"].sum(), points[-1])[
        #     0
        # ]  # .detach()
        points, meta = mala(
            points[-1],
            target,
            proposal,
            n_mala_steps,
            n_mala_steps - 1,
            project,
            step_size=step_size,
            target_acceptance=target_acceptance,
            meta=meta,
            keep_graph=keep_graph,
        )
        step_size = meta["step_size"][-1]
        point = points[-1]
        if not keep_graph:
            point = point.detach().requires_grad_()
        if step_id >= burn_in:
            chains.append(point)

    chains = torch.stack(chains, 0)
    return chains, meta


@MCMCRegistry.register()
def hmc(
    start: torch.FloatTensor,
    target: Union[Distribution, torchDist],
    proposal: Union[Distribution, torchDist],
    n_samples: int,
    burn_in: int,
    project: Callable,
    *,
    step_size: float,
    leapfrog_steps: int = 1,
    target_acceptance: float = False,
    adapt_step_size: bool = False,
    verbose: bool = False,
    meta: Optional[Dict] = None,
):
    meta = meta or dict()
    meta["step_size"] = meta.get("step_size", [])
    if "hmc_kernel" not in meta:
        kernel = HMC(
            potential_fn=lambda x: -target.log_prob(x["points"]),
            full_mass=False,
            step_size=step_size,
            num_steps=leapfrog_steps,
            adapt_step_size=adapt_step_size,
            target_accept_prob=target_acceptance,
        )
        meta["hmc_kernel"] = kernel

    x = start.clone()
    x.requires_grad_(False)
    x.grad = None

    init_params = {"points": x}
    mcmc = MCMC(
        kernel=meta["hmc_kernel"],
        num_samples=n_samples,
        initial_params=init_params,
        warmup_steps=burn_in,
    )
    mcmc.run()
    meta["step_size"].append(meta["hmc_kernel"].step_size)

    chains = mcmc.get_samples(group_by_chain=True)["points"]
    chains = chains.view(-1, *start.shape)

    return chains, meta


@MCMCRegistry.register()
def flex2mcmc(
    start: torch.FloatTensor,
    target,
    proposal,
    n_samples: int,
    burn_in: int,
    project: Callable,
    *,
    # proposal_opt,
    # proposal_scheduler,
    n_particles: int,
    step_size: float,
    n_mala_steps: int = 1,
    add_pop_size_train: int = 4096,
    forward_kl_weight: float = 1.0,
    backward_kl_weight: float = 1.0,
    target_acceptance: float = False,
    verbose: bool = False,
    meta: Optional[Dict] = None,
    keep_graph: bool = False,
) -> Tuple[torch.FloatTensor, Dict]:
    """
    Ex2MCMC with Flow proposal

    Args:
        start - strating points of shape [n_chains x dim]
        target - target distribution instance with method "log_prob"
        proposal - instance of RealNVProposal
        n_samples - number of last samples from each chain to return
        burn_in - number of first samples from each chain to throw away
        step_size - step size for drift term
        n_particles - number of particles including one from previous step
        n_mala_steps - number of MALA steps after each SIR step
        verbose - whether to show iterations' bar

    Returns:
        tensor of chains with shape [n_samples, n_chains, dim],
          acceptance rates for each iteration
    """
    chains = []
    if meta:
        meta["sir_accept"] = meta.get("sir_accept", [])
        meta["forward_kl"] = meta.get("forward_kl", [])
        meta["backward_kl"] = meta.get("backward_kl", [])
    else:
        meta = dict(sir_accept=[], forward_kl=[], backward_kl=[])

    x = start.clone()
    x.requires_grad_(True)
    x.grad = None

    hist_proposals = None

    pbar = trange(n_samples + burn_in) if verbose else range(n_samples + burn_in)
    for step_id in pbar:
        x, proposals, log_ps, _, indices = isir_step(
            x, target, proposal, n_particles=n_particles
        )
        meta["sir_accept"].append((indices != 0).float().mean().item())
        x, meta = mala(
            x,
            target,
            proposal.prior,
            n_mala_steps,
            n_mala_steps - 1,
            project,
            step_size=step_size,
            target_acceptance=target_acceptance,
            verbose=False,
            meta=meta,
        )
        step_size = meta["step_size"]
        x = x[-1]

        if step_id >= burn_in:
            chains.append(x)
        # else:
        if proposal.optim.param_groups[0]["lr"] > 0:
            # forward KL
            proposals_flattened = proposals.reshape(-1, x.shape[-1])
            log_ps_flattened = log_ps.reshape(-1).detach()

            if hist_proposals is not None:
                idxs = np.random.choice(hist_proposals.shape[0], add_pop_size_train)
                proposals_flattened = torch.cat(
                    (proposals_flattened, hist_proposals[idxs]), dim=0
                )
                log_ps_flattened = torch.cat(
                    (log_ps_flattened, hist_log_ps[idxs]), dim=0
                )

            log_qs = proposal.log_prob(proposals_flattened)
            logw = log_ps_flattened - log_qs.detach()

            kl_forw = -(log_qs * torch.softmax(logw, dim=0)).sum()

            # backward KL
            _, log_det_J = proposal.forward(proposals[:, 1:])
            kl_back = -(log_ps[:, 1:] + log_det_J).mean()

            # entropy reg
            e = -proposal.log_prob(proposal.prior.sample(proposals.shape)).mean()

            loss = forward_kl_weight * kl_forw + backward_kl_weight * kl_back + 0.1 * e
            proposal.optim.zero_grad()
            loss.backward()
            proposal.optim.step()
            proposal.scheduler.step()

            meta["forward_kl"].append(kl_forw.item())
            meta["backward_kl"].append(kl_back.item())

            if verbose:
                pbar.set_description(
                    f"KL forw {kl_forw.item():.3f}, \
                    KL back {kl_back.item():.3f} Hentr {e.item():.3f}"
                )
            print(
                f"KL forw {kl_forw.item():.3f}, \
                    KL back {kl_back.item():.3f} Hentr {e.item():.3f}"
            )

            if hist_proposals is None:
                hist_proposals = proposals_flattened
                hist_log_ps = log_ps_flattened
            else:
                hist_proposals = torch.cat((hist_proposals, proposals_flattened), dim=0)
                hist_log_ps = torch.cat((hist_log_ps, log_ps_flattened), dim=0)
    chains = torch.stack(chains, 0)
    return chains, meta
