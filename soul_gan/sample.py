from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from tqdm import trange

from .distribution import Distribution, grad_log_prob, torchDist
from .feature import Feature
from .utils import time_comp


def ula(
    z: torch.FloatTensor,
    target: Union[Distribution, torchDist],
    proposal: Union[Distribution, torchDist],
    step_size: float,
    n_steps: int = 1,
    project: Optional[Callable] = None,
) -> List[torch.FloatTensor]:
    zs = []
    grad_norms = []
    device = z.device

    for it in range(n_steps):
        _, grad = grad_log_prob(z, target)
        grad_norms.append(torch.norm(grad + z, dim=-1).mean().item())
        noise = torch.randn(z.shape, dtype=torch.float).to(device)
        noise_scale = (2.0 * step_size) ** 0.5
        z = z + step_size * grad + noise_scale * noise
        z = project(z)
        z = z.data
        z.requires_grad_(True)
        zs.append(z.data)

    return zs, grad_norms


@time_comp
def soul(
    z: torch.FloatTensor,
    gen: nn.Module,
    ref_dist: Union[Distribution, torchDist],
    feature: Feature,
    n_steps: int,
    burn_in_steps: int = 0,
    start_sample: int = 0,
    n_sampling_steps: int = 3,
    weight_step: float = 0.1,
    step_size: float = 0.01,
    save_every: int = 10,
) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor]]:
    # z.requires_grad_(True)
    zs = [z.cpu()]
    xs = [gen.inverse_transform(gen(z)).detach().cpu()]
    grad_norm = 0

    # saving parameter initialization
    # n_stride_im = params['stride_save_image']
    # n_stride_curve = params['stride_save_curve']
    # n_stride = min(n_stride_im, n_stride_curve)

    def target(z):
        f = feature(gen(z), z)
        radnic_logp = feature.log_prob(f)
        ref_logp = ref_dist(z)
        logp = radnic_logp + ref_logp
        return logp

    for it in trange(1, n_steps + 2):
        z.requires_grad_(True)
        condition_avg = it > burn_in_steps or it == 1
        condition_upd = it > burn_in_steps or it == 1

        if condition_upd:
            feature.weight_up(feature.avg_feature.data, weight_step, grad_norm)

        if condition_avg:
            feature.avg_weight.upd(feature.weight)

        feature.avg_feature.reset()

        inter_zs, grad_norms = ula(
            z,
            target,
            None,
            step_size,
            n_steps=n_sampling_steps,
            project=ref_dist.project,
        )
        if it > start_sample:
            z = inter_zs[-1]
            grad_norm = grad_norms[-1]

        if it % save_every == 0:
            zs.append(z.data.cpu())
            xs.append(gen.inverse_transform(gen(z)).data.cpu())

    return zs, xs
