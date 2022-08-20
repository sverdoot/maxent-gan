import copy
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from scipy.optimize import minimize  # noqa: F401
from torch import nn
from torch.distributions import Distribution as torchDist
from tqdm import trange

from maxent_gan.distribution import Distribution, MaxEntTarget
from maxent_gan.feature import BaseFeature
from maxent_gan.mcmc import MCMCRegistry
from maxent_gan.utils import time_comp_cls
from maxent_gan.utils.callbacks import Callback


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
        callbacks: Optional[Iterable[Callback]] = None,
        keep_graph: bool = False,
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
        self.callbacks = callbacks or []
        self.keep_graph = keep_graph

        self.sampling = sampling
        self.init_mcmc_args: Dict = copy.deepcopy(mcmc_args or dict())
        self.mcmc_args = copy.deepcopy(self.init_mcmc_args)

        self.radnic_logps = []
        self.ref_logps = []

        # if sweet_init:
        #     self.find_sweet_init()

        self.mcmc = MCMCRegistry()
        self.target = MaxEntTarget(gen, feature, ref_dist, batch_size=batch_size)

    def reset(self):
        self.mcmc_args = copy.deepcopy(self.init_mcmc_args)
        for callback in self.callbacks:
            callback.reset()

    @property
    def ref_dist(self):
        return self._ref_dist

    def step(
        self,
        z: torch.Tensor,
        it: int = 1,
        data_batch: Optional[torch.FloatTensor] = None,
        meta: Optional[Dict] = None,
        keep_graph: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        avg = (it > self.burn_in_steps or it == 1) and it % self.weight_avg_every == 0

        upd = (it > self.burn_in_steps or it == 1) and it % self.weight_upd_every == 0
        reset = (
            it > self.burn_in_steps or it == 1
        ) and it % self.feature_reset_every == 0

        self._ref_dist.data_batch = data_batch
        pts, meta = self.mcmc(
            self.sampling,
            z,
            self.target,
            proposal=self.gen.proposal,
            #n_samples=self.n_sampling_steps,
            #burn_in=self.n_sampling_steps - 1,
            project=self.ref_dist.project,
            **self.mcmc_args,
            meta=meta,
            keep_graph=keep_graph,
        )
        #print(self.feature.output_history[-1][0].mean(0), self.feature.ref_feature[0])

        self.mcmc_args.update(
            {key: meta[key][-1] for key in self.mcmc_args.keys() & meta.keys()}
        )
        self.feature.average_feature(meta["mask"])

        if upd:
            self.feature.weight_up(self.feature.avg_feature.data, self.weight_step)
        if reset:
            self.feature.avg_feature.reset()
        if avg:
            self.feature.avg_weight.upd(self.feature.weight)

        return pts[-1], meta

    @time_comp_cls
    def __call__(
        self,
        z: torch.Tensor,
        n_steps: Optional[int] = None,
        data_batch: Optional[torch.FloatTensor] = None,
        collect_imgs: bool = False,
        keep_graph: bool = False,
    ) -> Tuple[List, List, List, List]:
        n_steps = n_steps if n_steps is not None else self.n_steps
        collect_imgs = collect_imgs or self.collect_imgs
        keep_graph = keep_graph or self.keep_graph
        zs = [z.cpu()]
        xs = []
        meta = dict()
        if collect_imgs:
            xs.append(self.gen.inverse_transform(self.gen(z)).detach().cpu())
        self.target.radnic_logps = []
        self.target.ref_logps = []

        it = 0
        self.feature.avg_feature.reset()
        for it in self.trange(1, n_steps + 1):
            new_z, meta = self.step(z, it, data_batch, meta=meta, keep_graph=keep_graph)
            if it > self.start_sample:
                z = new_z

            if it > self.burn_in_steps and it % self.save_every == 0:
                if keep_graph:
                    zs.append(z.cpu())
                else:
                    zs.append(z.detach().cpu())
                if collect_imgs:
                    xs.append(
                        self.gen.inverse_transform(self.gen(z.detach())).detach().cpu()
                    )

            for callback in self.callbacks:
                callback.invoke(self.mcmc_args)
        self.feature.output_history = []

        # self.target.log_prob(z.detach(), data_batch) ??

        return zs, xs, self.target.ref_logps, self.target.radnic_logps

    # DON'T REMOVE ME
    # def find_sweet_init(
    #     self, n_steps: int = 100, batch_size: int = 256, burn_in: int = 50
    # ):
    #     z = self.gen.prior.sample((batch_size,))
    #     z.requires_grad_(True)
    #     device = z.device
    #     pts, meta = self.mcmc(
    #         self.sampling,
    #         z,
    #         self.target,
    #         proposal=None,
    #         n_samples=self.n_sampling_steps,
    #         burn_in=burn_in,
    #         project=self.ref_dist.project,
    #         **self.mcmc_args,
    #     )

    #     weights = self.feature.weight
    #     out = self.feature(self.gen(pts))
    #     self.feature.avg_feature.reset()
    #     out = [f_vals.detach() for f_vals in out]
    #     for i, (f_vals, weight) in enumerate(zip(out, weights)):
    #         rad_nic_p = torch.zeros(1)
    #         grad = torch.zeros(weight.shape)

    #         def objective(weight):
    #             weight = torch.from_numpy(weight).float().to(device)
    #             lik_f = -torch.einsum("ab,b->a", f_vals, weight)
    #             nonlocal rad_nic_p
    #             rad_nic_p = torch.exp(lik_f)
    #             return rad_nic_p.sum().cpu()

    #         def jac(weight):
    #             weight = torch.from_numpy(weight).float().to(device)
    #             nonlocal grad
    #             grad = -f_vals * rad_nic_p[:, None]
    #             return grad.sum(0).cpu()

    #         def hess(weight):
    #             weight = torch.from_numpy(weight).float().to(device)
    #             weight = [weight]
    #             sec = torch.einsum("ab,ac->abc", f_vals, grad)
    #             return sec.sum(0).cpu()

    #         x0 = weight.detach().cpu().numpy()
    #         res = minimize(objective, x0=x0, jac=jac, hess=hess, method="Newton-CG")
    #         weights[i] = torch.from_numpy(res.x).to(z.device)
    #     print(weights)
    #     self.feature.weights = weights
    #     return weights
