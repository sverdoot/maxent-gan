from typing import Iterable, Optional, Tuple

import numpy as np
import torch  # noqa: F401
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from maxent_gan.sample import MaxEntSampler
from maxent_gan.train.loss import Loss, gradient_penalty
from maxent_gan.utils.callbacks import Callback


class Trainer:
    def __init__(
        self,
        gan,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        criterion_g: Loss,
        criterion_d: Loss,
        dataloader: DataLoader,
        device,
        *,
        start_iter: int = 1,
        eval_every: int = 1000,
        n_dis: int = 1,
        n_gen: int = 1,
        gp_coef: float = 0.0,
        sample_size: int = 1000,
        grad_acc_steps: int = 1,
        sample_steps: int = 0,
        sampler: Optional[MaxEntSampler] = None,
        callbacks: Optional[Iterable[Callback]] = None,
        **kwargs,
    ):
        self.gan = gan
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.criterion_g = criterion_g
        self.criterion_d = criterion_d
        self.dataloader = dataloader
        self.device = device
        self.n_dis = n_dis
        self.n_gen = n_gen
        self.gp_coef = gp_coef
        self.sample_size = sample_size
        self.grad_acc_steps = grad_acc_steps
        self.sample_steps = sample_steps
        self.sampler = sampler
        self.callbacks = callbacks if callbacks else []
        self.start_iter = start_iter
        self.eval_every = eval_every

        self.replay_buffer = []

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
        prior = self.gan.gen.prior
        d_batch = real_batch

        self.optimizer_d.zero_grad()
        self.freeze_gen()
        self.unfreeze_dis()
        loss_d_num = 0
        for step_id in range(1, self.n_dis * self.grad_acc_steps + 1):
            latent = prior.sample((d_batch.shape[0],))
            if self.sampler:
                latents, _, _, _ = self.sampler(latent, keep_graph=True)
                latent = latents[-1]
            fake_batch = self.gan.gen(latent.to(self.device))
            score_fake = self.gan.dis(fake_batch).squeeze()
            score_real = self.gan.dis(d_batch).squeeze()

            loss_d = self.criterion_d(score_fake, score_real)
            if self.gp_coef > 0:
                loss_d += gradient_penalty(
                    self.gan.dis, d_batch, fake_batch, self.gp_coef
                )
            loss_d /= self.grad_acc_steps
            loss_d.backward()
            loss_d_num += loss_d.item()
            if step_id % self.grad_acc_steps == 0:
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()

        self.freeze_dis()
        self.unfreeze_gen()
        self.optimizer_g.zero_grad()
        loss_g_num = 0
        for step_id in range(1, self.n_gen * self.grad_acc_steps + 1):
            latent = prior.sample((d_batch.shape[0],))
            if self.sampler:
                latents, _, _, _ = self.sampler(latent, keep_graph=True)
                latent = latents[-1]
            fake_batch = self.gan.gen(latent.to(self.device))
            fake_scores = self.gan.dis(fake_batch).squeeze()

            loss_g = self.criterion_g(fake_scores)
            loss_g /= self.grad_acc_steps
            loss_g.backward()
            loss_g_num += loss_g.item()

            if step_id % self.grad_acc_steps == 0:
                self.optimizer_g.step()
                self.optimizer_g.zero_grad()

        return loss_g_num, loss_d_num

    def train(self):
        self.gan.dis.train()
        self.gan.gen.train()
        loss_g, loss_d = 0, 0
        for batch_id, batch in tqdm(
            enumerate(self.dataloader, 1), total=len(self.dataloader)
        ):
            if batch_id < self.start_iter:
                continue
            batch = batch.to(self.device)
            l_g, l_d = self.step(batch)
            loss_g += l_g / self.eval_every
            loss_d += l_d / self.eval_every

            if batch_id % self.eval_every == 0:
                self.gan.gen.eval()
                imgs = []
                sample_size = self.sample_size
                while sample_size > 0:
                    fake_batch = self.gan.gen(
                        self.gan.gen.prior.sample((min(sample_size, batch.shape[0]),))
                    )
                    imgs.append(
                        self.gan.gen.inverse_transform(fake_batch)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    sample_size -= batch.shape[0]
                imgs = np.concatenate(imgs, axis=0)

                if self.sampler:
                    weight = self.sampler.feature.weight
                    weight_norm = weight[0].norm().item() if len(weight) > 0 else 0
                else:
                    weight_norm = 0
                info = dict(
                    step=batch_id,
                    total=len(self.dataloader),
                    loss_g=loss_g,
                    loss_d=loss_d,
                    imgs=imgs,
                    weight_norm=weight_norm,
                )
                self.gan.gen.train()
                loss_g, loss_d = 0, 0
            else:
                info = dict(step=batch_id)
            for callback in self.callbacks:
                callback.invoke(info)
