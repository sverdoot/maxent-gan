from typing import Dict, List, Optional, Tuple, Union

import torch
from scipy.optimize import minimize
from torch import nn
from torch.distributions import Distribution as torchDist
from tqdm import trange

from .distribution import Distribution, MaxEntTarget
from .feature import BaseFeature
from .mcmc import MCMCRegistry
from .utils import time_comp_cls


class MaxEntSampler:
    def __init__(
        self,
        gen: nn.Module,
        ref_dist: Union[Distribution, torchDist],
        feature: BaseFeature,
        *,
        burn_in_steps: int = 0,
        start_sample: int = 0,
        n_sampling_steps: int = 1,
        weight_step: float = 0.1,
        # step_size: float = 0.01,
        batch_size: Optional[int] = None,
        save_every: int = 1,
        verbose: bool = True,
        n_steps: Optional[int] = None,
        weight_upd_every: int = 1,
        weight_avg_every: int = 1,
        feature_reset_every: int = 1,
        sweet_init: bool = False,
        collect_imgs: bool = True,
        sampling: str = "ula",
        mcmc_args: Optional[Dict] = None,
    ):
        self.gen = gen
        self._ref_dist = ref_dist
        self.feature = feature
        self.burn_in_steps = burn_in_steps
        self.start_sample = start_sample
        self.n_sampling_steps = n_sampling_steps
        self.weight_step = weight_step
        self.save_every = save_every
        self.verbose = verbose
        self.trange = trange if self.verbose else range
        self.n_steps = n_steps
        self.weight_upd_every = weight_upd_every
        self.weight_avg_every = weight_avg_every
        self.feature_reset_every = feature_reset_every
        self.collect_imgs = collect_imgs

        self.sampling = sampling
        self.mcmc_args: Dict = mcmc_args or dict()

        self.radnic_logps = []
        self.ref_logps = []

        if sweet_init:
            self.find_sweet_init()

        self.mcmc = MCMCRegistry()
        self.target = MaxEntTarget(gen, feature, ref_dist, batch_size=batch_size)

    @property
    def ref_dist(self):
        return self._ref_dist

    def step(
        self,
        z: torch.Tensor,
        it: int = 1,
        data_batch: Optional[torch.FloatTensor] = None,
        meta=None,
    ) -> Tuple[torch.Tensor, Dict]:
        z.requires_grad_(True)
        avg = (it > self.burn_in_steps or it == 1) and it % self.weight_avg_every == 0

        upd = (it > self.burn_in_steps or it == 1) and it % self.weight_upd_every == 0
        reset = (
            it > self.burn_in_steps or it == 1
        ) and it % self.feature_reset_every == 0

        if upd:
            self.feature.weight_up(
                self.feature.avg_feature.data, self.weight_step
            )  # , grad_norm)
        if reset:
            self.feature.avg_feature.reset()

        if avg:
            self.feature.avg_weight.upd(self.feature.weight)

        proposal = self.gen.prior
        if isinstance(proposal, torch.distributions.MultivariateNormal):
            proposal = torch.distributions.MultivariateNormal(
                torch.zeros(z.shape[-1]).to(z.device),
                2 * torch.eye(z.shape[-1]).to(z.device),
            )

        self._ref_dist.data_batch = data_batch
        pts, meta = self.mcmc(
            self.sampling,
            z,
            self.target,
            proposal=proposal,
            n_samples=self.n_sampling_steps,
            burn_in=0,
            project=self.ref_dist.project,
            **self.mcmc_args,
            meta=meta,
        )

        for key in self.mcmc_args.keys() & meta.keys():
            self.mcmc_args.update(key, meta[key][-1])

        return pts[-1], meta

    @time_comp_cls
    def __call__(
        self,
        z: torch.Tensor,
        n_steps: Optional[int] = None,
        data_batch: Optional[torch.FloatTensor] = None,
        collect_imgs: Optional[bool] = None,
    ) -> Tuple[List, List, List, List]:
        n_steps = n_steps or self.n_steps
        collect_imgs = collect_imgs or self.collect_imgs
        zs = [z.cpu()]
        xs = []
        meta = dict()
        if collect_imgs:
            xs.append(self.gen.inverse_transform(self.gen(z)).detach().cpu())
        self.target.radnic_logps = []
        self.target.ref_logps = []

        it = 0
        self.feature.avg_feature.reset()
        for it in self.trange(1, n_steps + 2):
            new_z, meta = self.step(z, it, data_batch, meta=meta)
            if it > self.start_sample:
                z = new_z

            if it > self.burn_in_steps and it % self.save_every == 0:
                zs.append(z.data.cpu())
                if collect_imgs:
                    xs.append(self.gen.inverse_transform(self.gen(z)).data.cpu())

        self.target.log_prob(z, data_batch)

        return zs, xs, self.target.ref_logps, self.target.radnic_logps

    def find_sweet_init(
        self, n_steps: int = 100, batch_size: int = 256, burn_in: int = 50
    ):
        z = self.gen.prior.sample((batch_size,))
        z.requires_grad_(True)
        device = z.device
        pts, meta = self.mcmc(
            self.sampling,
            z,
            self.target,
            proposal=None,
            n_samples=self.n_sampling_steps,
            burn_in=burn_in,
            project=self.ref_dist.project,
            **self.mcmc_args,
        )

        weights = self.feature.weight
        out = self.feature(self.gen(pts))
        self.feature.avg_feature.reset()
        out = [f_vals.detach() for f_vals in out]
        for i, (f_vals, weight) in enumerate(zip(out, weights)):
            rad_nic_p = torch.zeros(1)
            grad = torch.zeros(weight.shape)

            def objective(weight):
                weight = torch.from_numpy(weight).float().to(device)
                lik_f = -torch.einsum("ab,b->a", f_vals, weight)
                nonlocal rad_nic_p
                rad_nic_p = torch.exp(lik_f)
                return rad_nic_p.sum().cpu()

            def jac(weight):
                weight = torch.from_numpy(weight).float().to(device)
                nonlocal grad
                grad = -f_vals * rad_nic_p[:, None]
                return grad.sum(0).cpu()

            def hess(weight):
                weight = torch.from_numpy(weight).float().to(device)
                weight = [weight]
                sec = torch.einsum("ab,ac->abc", f_vals, grad)
                return sec.sum(0).cpu()

            x0 = weight.detach().cpu().numpy()
            res = minimize(objective, x0=x0, jac=jac, hess=hess, method="Newton-CG")
            weights[i] = torch.from_numpy(res.x).to(z.device)
        print(weights)
        self.feature.weights = weights
        return weights
