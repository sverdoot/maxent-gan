from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from maxent_gan.sample import MaxEntSampler
from maxent_gan.utils.callbacks import Callback
from maxent_gan.utils.train.loss import Loss, gradient_penalty


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
        start_epoch: int = 0,
        n_dis: int = 1,
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
        self.gp_coef = gp_coef
        self.sample_size = sample_size
        self.grad_acc_steps = grad_acc_steps
        self.sample_steps = sample_steps
        self.sampler = sampler
        self.callbacks = callbacks if callbacks else []
        self.start_epoch = start_epoch

        self.replay_buffer = []

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
        prior = self.gan.gen.prior
        batch_size = real_batch.shape[0]
        # d_batch = real_batch
        d_batch = real_batch[: batch_size // 2]

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
            loss_d_num = loss_d.item()
            if step_id % self.grad_acc_steps == 0:
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()

        self.gan.dis.zero_grad()
        self.freeze_dis()
        self.unfreeze_gen()
        self.optimizer_g.zero_grad()
        for step_id in range(1, self.grad_acc_steps + 1):
            loss_g_num, loss_g = 0, 0
            latent = prior.sample((d_batch.shape[0],))
            if self.sampler:
                latents, _, _, _ = self.sampler(latent, keep_graph=True)
                latent = latents[-1]
            fake_batch = self.gan.gen(latent.to(self.device))
            fake_scores = self.gan.dis(fake_batch).squeeze()

            loss_g += self.criterion_g(fake_scores)
            loss_g /= self.grad_acc_steps
            loss_g.backward()
            loss_g_num += loss_g.item()

        # torch.nn.utils.clip_grad_norm_(self.gan.gen.parameters(), 20.0)
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()

        return loss_g_num, loss_d_num

    def train(self, n_epochs):
        batch_size = None
        for ep in range(self.start_epoch, n_epochs + 1):
            self.gan.gen.train()

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
            imgs = []
            sample_size = self.sample_size
            while sample_size > 0:
                fake_batch = self.gan.gen(
                    self.gan.gen.prior.sample((min(sample_size, batch_size),))
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
                loss_g=loss_g,
                loss_d=loss_d,
                imgs=imgs,
                weight_norm=weight_norm,
            )
            for callback in self.callbacks:
                callback.invoke(info)
