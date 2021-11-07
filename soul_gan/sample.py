import numpy as np
import scipy as sp
import torch
from easydict import EasyDict as edict
from tqdm import tqdm, trange

from .distribution import grad_energy
from .utils import time_comp


def ula(z, target, proposal, step_size, n_steps=1):
    zs = []
    device = z.device
    
    for it in range(n_steps):
        energy, grad = grad_energy(z, target)
        noise = torch.randn(z.shape, dtype=torch.float).to(device)
        noise_scale = (step_size)**.5
        z = z - step_size * grad + noise_scale * noise
        z = z.data
        z.requires_grad_(True)
        zs.append(z)

    return zs


@time_comp
def soul(z, gen, ref_dist, feature, params):
    zs = []
    params = edict(params)

    # saving parameter initialization
    #n_stride_im = params['stride_save_image']
    #n_stride_curve = params['stride_save_curve']
    #n_stride = min(n_stride_im, n_stride_curve)
    ne = 0  # number of epochs

    feature.weight = params.weight
    feature(gen(z))

    def target(z):
        f = feature(gen(z))
        radnic_logp = feature.log_prob(f)
        ref_logp = ref_dist(z).sum()
        logp = radnic_logp + ref_logp
        return logp

    # store initialization
    # saving folder initialization
    #fd = save_init(params, feature)

    for it in trange(params.n_steps):
        #cond = params['save'] == 'all' and (np.mod(it, n_stride) == 0)
        z.requires_grad_()

        # WEIGHT UPDATE
        with torch.no_grad():
            feature.weight = feature.weight_up(feature.avg_feature.data, params.weight_step)
            condition_avg = it > params.burnin or it == 0

            # weight average
            if condition_avg:
                n_avg = max(it - params.burnin, 0)
                feature.avg_weight.upd(feature.weight)

            # save weight
            #if cond and condition_avg:
            #    feature.save_weight(params, f_avg, w, w_avg, it, ne, fd)
        feature.avg_feature.reset()

        ne += params.n_sampling_steps
        z = ula(z, target, None, params.step_size, n_steps=params.n_sampling_steps)
        zs.append(z)     
    # build final model
    #model = build_model(params, x, x_grad, x0, w, w_avg, w0, feature)
    return zs
