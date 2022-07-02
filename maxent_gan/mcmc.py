from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pyro.infer import HMC, MCMC
from torch.distributions import Categorical
from torch.distributions import Distribution as torchDist
from torch.distributions import Normal
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
    meta: Optional[Dict] = None,
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
            chains.append(point.detach().data.clone())
    chains = torch.stack(chains, 0)

    meta["logp"] = logp_x
    meta["mask"] = F.one_hot(indices, num_classes=n_particles).to(bool)

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
    x = start.clone()
    x.requires_grad_()
    x.grad = None

    meta = meta or dict()
    meta["mh_accept"] = meta.get("mh_accept", [])
    meta["step_size"] = meta.get("step_size", [])
    # meta["logp"] = meta.get("logp", target.log_prob(x))
    # meta["grad"] = meta.get("grad", torch.autograd.grad(logp_x.sum(), x)[0])

    if "grad" not in meta:
        logp_x = target.log_prob(x)
        grad_x = torch.autograd.grad(logp_x.sum(), x)[0]
        meta["logp"] = logp_x
        meta["grad"] = grad_x
    logp_x = meta["logp"]
    grad_x = meta["grad"]

    range_ = trange if verbose else range
    for step_id in range_(n_samples + burn_in):
        noise = proposal.sample(x.shape[:-1])
        y = x + step_size * grad_x + noise * (2 * step_size) ** 0.5
        y = project(y)

        y = y.detach().requires_grad_()
        logp_y = target.log_prob(y)
        grad_y = torch.autograd.grad(logp_y.sum(), y)[0]

        log_qyx = proposal.log_prob(noise)
        log_qxy = proposal.log_prob(
            (x - y - step_size * grad_y) / (2 * step_size) ** 0.5
        )

        accept_prob = torch.clamp((logp_y + log_qxy - logp_x - log_qyx).exp(), max=1)
        mask = torch.rand_like(accept_prob) < accept_prob

        with torch.no_grad():
            x[mask, :] = y[mask, :]
            logp_x[mask] = logp_y[mask]
            grad_x[mask] = grad_y[mask]

        meta["mh_accept"].append(mask.float().mean().item())
        if target_acceptance:
            step_size = heuristics_step_size(
                meta["mh_accept"][-1], target_acceptance, step_size
            )
        meta["step_size"].append(step_size)

        if step_id >= burn_in:
            chains.append(x.detach().data.clone())
    chains = torch.stack(chains, 0)

    meta["logp"] = logp_x
    meta["grad"] = grad_x
    meta["mask"] = mask

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
    meta: Optional[Dict] = None,
):
    chains = []
    x = start.clone()
    x.requires_grad_(True)
    x.grad = None

    meta = meta or dict()
    meta["sir_accept"] = meta.get("sir_accept", [])
    meta["mh_accept"] = meta.get("mh_accept", [])
    meta["step_size"] = meta.get("step_size", [])

    range_ = trange if verbose else range
    for _ in range_(n_samples + burn_in):
        xs, meta = isir(
            x, target, proposal, 1, 0, project, n_particles=n_particles, meta=meta
        )
        meta["grad"] = torch.autograd.grad(meta["logp"].sum(), x)[0]
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
        step_size = meta["step_size"][-1]
        x = xs[1].data
        x.requires_grad_()
        chains.append(x.clone().detach())

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
    target_acceptance_rate: float = False,
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
            target_accept_prob=target_acceptance_rate,
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
    n_particles: int,
    step_size: float,
    n_mala_steps: int,
    add_pop_size_train: int = 4096,
    forward_kl_weight: float = 1.0,
    backward_kl_weight: float = 1.0,
    meta: Optional[Dict] = None,
    verbose: bool = False,
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
        tensor of chains with shape [n_samples, n_chains, dim], acceptance rates for each iteration
    """
    chains = []
    if meta:
        meta["sir_accept"] = meta.get("sir_accept", [])
        meta["forward_kl"] = meta.get("forward_kl", [])
        meta["backward_kl"] = meta.get("backward_kl", [])
    else:
        meta = dict(sir_accept=[], forward_kl=[], backward_kl=[])
    meta["proposal_opt"] = meta.get(
        "proposal_opt", torch.optim.Adam(proposal.parameters(), lr=1e-3)
    )

    x = start.clone()
    x.requires_grad_(True)
    x.grad = None

    hist_proposals = None

    pbar = trange(n_samples + burn_in) if verbose else range(n_samples + burn_in)
    for step_id in pbar:
        x = x.data
        x.requires_grad_(True)

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
            verbose=False,
            meta=meta,
        )
        step_size = meta["step_size"]
        x = x[-1]

        if step_id >= burn_in:
            chains.append(x.detach().data.clone())
        else:
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
            meta["proposal_opt"].zero_grad()
            loss.backward()
            meta["proposal_opt"].step()

            meta["forward_kl"].append(kl_forw.item())
            meta["backward_kl"].append(kl_back.item())

            if verbose:
                pbar.set_description(
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
