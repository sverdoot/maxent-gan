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
    device = z.device

    for it in range(n_steps):
        _, grad = grad_log_prob(z, target)
        noise = torch.randn(z.shape, dtype=torch.float).to(device)
        noise_scale = (2.0 * step_size) ** 0.5
        z = z + step_size * grad + noise_scale * noise
        z = project(z)
        z = z.data
        z.requires_grad_(True)
        zs.append(z.data)

    return zs


@time_comp
def soul(
    z: torch.FloatTensor,
    gen: nn.Module,
    ref_dist: Union[Distribution, torchDist],
    feature: Feature,
    n_steps: int,
    burn_in_steps: int,
    n_sampling_steps: int = 3,
    weight_step: float = 0.1,
    step_size: float = 0.01,
    save_every: int = 10,
) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor]]:
    zs = [z.cpu()]
    xs = [gen.inverse_transform(gen(z)).detach().cpu()]

    # saving parameter initialization
    # n_stride_im = params['stride_save_image']
    # n_stride_curve = params['stride_save_curve']
    # n_stride = min(n_stride_im, n_stride_curve)

    def target(z):
        f = feature(gen(z))
        radnic_logp = feature.log_prob(f)
        ref_logp = ref_dist(z)
        # print(ref_logp.mean(), radnic_logp.mean())
        logp = radnic_logp + ref_logp
        return logp


    for it in trange(1, n_steps + 2):
        # cond = params['save'] == 'all' and (np.mod(it, n_stride) == 0)
        z.requires_grad_()

        with torch.no_grad():
            condition_avg = it > burn_in_steps or it == 1
            condition_upd = it > burn_in_steps or it == 1
            if condition_upd:
                feature.weight_up(feature.avg_feature.data, weight_step)

            if condition_avg:
                # n_avg = max(it - burn_in_steps, 0)
                feature.avg_weight.upd(feature.weight)

            # save weight
            # if cond and condition_avg:
            #    feature.save_weight(params, f_avg, w, w_avg, it, ne, fd)
        feature.avg_feature.reset()

        inter_zs = ula(
            z,
            target,
            None,
            step_size,
            n_steps=n_sampling_steps,
            project=ref_dist.project,
        )
        z = inter_zs[-1]

        if it % save_every == 0:
            zs.append(z.detach().cpu())
            xs.append(gen.inverse_transform(gen(z)).detach().cpu())

    return zs, xs
