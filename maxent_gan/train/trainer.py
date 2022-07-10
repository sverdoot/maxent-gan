from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from maxent_gan.distribution import CondTarget, DiscriminatorTarget, PriorTarget
from maxent_gan.sample import MaxEntSampler
from maxent_gan.train.loss import gradient_penalty
from maxent_gan.utils.callbacks import Callback


# def opt_weight(gan, flow, feature, batch_size):
#     z = gan.gen.prior.sample((batch_size,))
#     z.requires_grad_(True)
#     device = z.device

#     weights = feature.weight
#     x = gan.gen(flow(z)[0])
#     scores = gan.dis(x).detach().squeeze()
#     out = [feature.apply(x)[0] - feature.ref_feature[0].to(device)[None, :]]
#     out = [f_vals.detach() for f_vals in out]
#     for i, (f_vals, weight) in enumerate(zip(out, weights)):
#         importance_weight = torch.zeros(1)
#         norm_weights = torch.zeros(1)
#         m = torch.zeros(0)
#         grad = torch.zeros(weight.shape)

#         def objective(weight):
#             weight = torch.from_numpy(weight).float().to(device)
#             lik_f = -torch.einsum("ab,b->a", f_vals, weight)
#             nonlocal importance_weight
#             importance_weight = torch.exp(lik_f + scores)
#             nonlocal norm_weights
#             norm_weights = torch.softmax(lik_f + scores, 0)
#             nonlocal m
#             m = importance_weight.mean().item()
#             return np.log(m)

#         def jac(weight):
#             weight = torch.from_numpy(weight).float().to(device)
#             nonlocal grad
#             grad = -f_vals * norm_weights[:, None]
#             return grad.mean(0).cpu()

#         def hess(weight):
#             weight = torch.from_numpy(weight).float().to(device)
#             weight = [weight]
#             sec = torch.einsum("ab,ac->abc", f_vals, grad).mean(0) - torch.einsum(
#                 "b,c->bc", grad.mean(0), grad.mean(0)
#             )
#             return sec.cpu()

#         lik_f = -torch.einsum("ab,b->a", f_vals, weight)
#         norm_weights = torch.softmax(lik_f + scores, 0)
#         weights[i].data -= 0.001 * (-f_vals * norm_weights[:, None]).mean(0)

#         # w0 = np.zeros(weight.shape) #f_vals.mean(0).cpu().numpy() #weight.detach().cpu().numpy()
#         # #w0 = f_vals.mean(0).cpu().numpy()
#         # #w0 = weight.detach().cpu().numpy()
#         # res = minimize(objective, x0=w0, jac=jac,
#         #     #hess=hess, method="Newton-CG",
#         #     options={"maxiter": 5})
#         # weights[i] = torch.from_numpy(res.x).to(z.device).float()
#     # feature.weights = weights
#     # print('ll', torch.einsum("ab,b->a", f_vals, weights[0]).mean(0))

#     return weights


class FlowIdentity(nn.Identity):
    def forward(self, x):
        return x, torch.ones(x.shape[0])


class OptimIdentity(Optimizer):
    def __init__(self) -> None:
        pass

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        pass

    def zero_grad(self, set_to_none: Optional[bool] = ...) -> None:
        pass


class Trainer:
    def __init__(
        self,
        gan,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        criterion_g,
        criterion_d,
        dataloader: DataLoader,
        device,
        *,
        flow: Optional[nn.Module] = None,
        optimizer_flow: Optional[Optimizer] = None,
        start_epoch: int = 0,
        n_dis: int = 1,
        gp_coef: float = 0.0,
        sample_size: int = 1000,
        grad_acc_steps: int = 1,
        sample_steps: int = 0,
        sampler: Optional[MaxEntSampler] = None,
        callbacks: Optional[Iterable[Callback]] = None,
        importance_sampling: bool = False,
        alpha: float = 10,
    ):
        self.gan = gan
        if not flow:
            flow = FlowIdentity()
            optimizer_flow = OptimIdentity()
        self.flow = flow
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.optimizer_flow = optimizer_flow
        self.criterion_g = criterion_g
        self.criterion_d = criterion_d
        self.dataloader = dataloader
        self.device = device
        self.n_dis = n_dis
        self.gp_coef = gp_coef
        self.sample_size = sample_size
        self.grad_acc_steps = grad_acc_steps
        self.sample_steps = sample_steps
        self.sampler = sampler
        self.callbacks = callbacks if callbacks else []
        self.importance_sampling = importance_sampling
        self.alpha = alpha
        self.start_epoch = start_epoch

    def sample_fake(self, batch_size: int, sample_fn: Optional[Callable] = None):
        latent = self.gan.gen.prior.sample((batch_size,))
        if sample_fn:
            latents = sample_fn(latent)[0]
            fake_batch = self.gan.gen(latents[-1].to(self.device))
        else:
            fake_batch = self.gan.gen(latent)
        return fake_batch

    def freeze_gen(self):
        for p in self.gan.gen.parameters():
            p.requires_grad_(False)

    def unfreeze_gen(self):
        for p in self.gan.gen.parameters():
            p.requires_grad_(True)

    def freeze_dis(self):
        for p in self.gan.dis.parameters():
            p.requires_grad_(False)

    def unfreeze_dis(self):
        for p in self.gan.dis.parameters():
            p.requires_grad_(True)

    def step(self, real_batch) -> Tuple[float, float]:
        # weight_norm = self.sampler.feature.weight[0].norm().item() if self.sampler else 0
        prior = self.gan.gen.prior
        batch_size = real_batch.shape[0]
        d_batch = real_batch[: batch_size // 2]
        g_batch = real_batch[batch_size // 2 :]

        self.optimizer_d.zero_grad()
        self.freeze_gen()
        self.unfreeze_dis()
        loss_d_num = 0
        for step_id in range(1, self.n_dis * self.grad_acc_steps + 1):
            latent = prior.sample((d_batch.shape[0],))
            latent, _ = self.flow(latent)
            fake_batch = self.gan.gen(latent.detach().to(self.device))
            score_fake = self.gan.dis(fake_batch).squeeze()
            weight = None

            score_real = self.gan.dis(d_batch).squeeze()
            loss_d = self.criterion_d(score_fake, score_real, weight=weight)
            if self.gp_coef > 0:
                loss_d += gradient_penalty(
                    self.gan.dis, d_batch, fake_batch, self.gp_coef
                )
            loss_d /= self.grad_acc_steps
            loss_d.backward()
            loss_d_num = loss_d.item()
            if step_id % self.grad_acc_steps == 0:
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()

        self.gan.dis.zero_grad()
        self.freeze_dis()
        self.unfreeze_gen()
        self.optimizer_g.zero_grad()
        for step_id in range(1, self.grad_acc_steps + 1):
            loss_g_num, loss_g, weight = 0, 0, None

            z0 = prior.sample((d_batch.shape[0],))
            z_push, logdetjac = self.flow(z0)

            if self.sampler:
                self.gan.gen.zero_grad()
                if self.sampler and not self.importance_sampling:
                    fake_batch = self.gan.gen(z_push)
                    scores = self.gan.dis(fake_batch).squeeze()
                else:
                    latents, _, ref_logps, radnic_logps = self.sampler(
                        self.soul_latent, data_batch=g_batch
                    )
                    self.soul_latent = prior.sample((g_batch.shape[0],))
                    ref_logps = torch.cat(ref_logps, 0)
                    radnic_logps = torch.cat(radnic_logps, 0)
                    latents_cat = torch.cat(latents, 0)
                    if isinstance(self.sampler.ref_dist, CondTarget):
                        weight = torch.softmax(
                            -radnic_logps.to(self.device) / self.alpha, dim=0
                        ).data * len(latents_cat)
                    elif isinstance(
                        self.sampler.ref_dist, (PriorTarget, DiscriminatorTarget)
                    ):
                        log_priors = prior.log_prob(latents_cat.to(self.device))
                        weight = (
                            torch.softmax(
                                (-ref_logps - radnic_logps + log_priors).to(self.device)
                                / self.alpha,
                                dim=0,
                            ).data
                            * len(latents_cat)
                        )
                    else:
                        raise ValueError(
                            f"Unknown reference distribution: {type(self.sampler.ref_dist)}"
                        )
                    scores = []
                    for lat in latents:
                        scores.append(
                            self.gan.dis(
                                self.gan.gen(lat.detach().to(self.device))
                            ).squeeze()
                        )
                    scores = torch.cat(scores, 0)
            else:
                fake_batch = self.gan.gen(z_push.to(self.device))
                scores = self.gan.dis(fake_batch).squeeze()

            if not isinstance(self.flow, FlowIdentity):
                feat = self.sampler.feature.apply_and_shift(fake_batch)
                self.sampler.feature.weight[0].data -= self.sampler.weight_step * (
                    -feat[0].mean(0) + 0.01 * self.sampler.feature.weight[0].data
                )
                # self.sampler.feature.weight_up([f.mean(0).detach() for f in feat],
                # self.sampler.weight_step)
                lik_f = self.sampler.feature.log_prob(feat).squeeze()
                # (torch.logsumexp(lik_f, dim=0) / len(lik_f)).backward(retain_graph=True)
                flow_kl_forward = (
                    -logdetjac
                    - self.sampler.ref_dist(z_push)
                    - lik_f
                    + prior.log_prob(z0)
                ).mean(0)
                loss_flow = flow_kl_forward
                self.optimizer_flow.zero_grad()
                loss_flow.backward(retain_graph=True)
                print(
                    flow_kl_forward,
                    logdetjac.mean(),
                    prior.log_prob(z_push).mean(),
                    lik_f.mean(),
                    prior.log_prob(z0).mean(),
                )
                print(feat[0].mean(0).norm(dim=0))
                print("w", self.sampler.feature.weight[0][0])

            loss_g += self.criterion_g(scores, weight=weight)
            loss_g /= self.grad_acc_steps
            flow_grads = dict()
            for n, p in self.flow.named_parameters():
                flow_grads[n] = torch.clone(p.grad).detach()
            loss_g.backward()
            for n, p in self.flow.named_parameters():
                p.grad = flow_grads[n]
            loss_g_num += loss_g.item()
        torch.nn.utils.clip_grad_norm_(self.gan.gen.parameters(), 20.0)
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()

        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 20.0)
        self.optimizer_flow.step()
        self.optimizer_flow.zero_grad()

        return loss_g_num, loss_d_num

    def train(self, n_epochs):
        batch_size = None
        # self.soul_latent = self.gan.prior.sample((32,))
        for ep in range(self.start_epoch, n_epochs + 1):
            self.gan.gen.train()
            self.flow.train()

            loss_g, loss_d = 0, 0
            from tqdm import tqdm

            for batch in tqdm(self.dataloader):
                if not batch_size:
                    batch_size = batch.shape[0]
                batch = batch.to(self.device)
                l_g, l_d = self.step(batch)
                loss_g += l_g / len(self.dataloader)
                loss_d += l_d / len(self.dataloader)

            self.gan.gen.eval()
            self.flow.eval()
            imgs = []
            sample_size = self.sample_size
            while sample_size > 0:
                fake_batch = self.gan.gen(
                    self.flow(
                        self.gan.gen.prior.sample((min(sample_size, batch_size),))
                    )[0]
                )
                imgs.append(
                    self.gan.gen.inverse_transform(fake_batch).detach().cpu().numpy()
                )
                sample_size -= batch_size

            imgs = np.concatenate(imgs, axis=0)
            if self.sampler:
                weight = self.sampler.feature.weight
                weight_norm = weight[0].norm().item() if len(weight) > 0 else 0
            else:
                weight_norm = 0
            info = dict(
                step=ep,
                total=n_epochs,
                loss_g=loss_g,
                loss_d=loss_d,
                imgs=imgs,
                weight_norm=weight_norm,
            )
            for callback in self.callbacks:
                callback.invoke(info)
