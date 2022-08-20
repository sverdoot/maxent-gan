from typing import Iterable, Optional, Tuple

import numpy as np
import torch  # noqa: F401
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from maxent_gan.sample import MaxEntSampler
from maxent_gan.train.loss import Loss, gradient_penalty, r1_penalty
from maxent_gan.utils.callbacks import Callback


class ReplayBuffer:
    def __init__(self, size: int, dim: int):
        self.size = size
        self.buffer = torch.empty((0, dim))

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, items: torch.Tensor):
        self.buffer = torch.cat([items.detach().cpu(), self.buffer], 0)
        self.free()

    def free(self):
        #self.buffer = self.buffer[torch.randperm(len(self))][:self.size]
        self.buffer = self.buffer[:self.size]

    def pop(self, batch_size) -> torch.Tensor:
        #ids = np.random.choice(np.arange(len(self)), batch_size)
        return self.buffer[:batch_size]


class DumbSampler:
    mcmc_args: dict = {}
    def __call__(self, x, *args, **kwargs):
        return [x], None, None, None


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
        r1_coef: float = 0.0,
        sample_size: int = 10000,
        grad_acc_steps: int = 1,
        sample_steps: int = 0,
        sampler: Optional[MaxEntSampler] = None,
        callbacks: Optional[Iterable[Callback]] = None,
        replay_prob: float = 0.0,
        replay_size: int = 0,
        meta_rate: float = 0.0,
        step_every: int =  1,
        # clip_grad_norm: float = float('inf'),
        **kwargs,
    ):
        self.gan = gan
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.criterion_g = criterion_g
        self.criterion_d = criterion_d
        # self.clip_grad_norm = clip_grad_norm
        self.dataloader = dataloader
        self.device = device
        self.n_dis = n_dis
        self.n_gen = n_gen
        self.gp_coef = gp_coef
        self.r1_coef = r1_coef
        self.sample_size = sample_size
        self.grad_acc_steps = grad_acc_steps
        self.sample_steps = sample_steps
        self.replay_prob = replay_prob if sampler.n_steps > 0 else 0.0
        self.replay_size = replay_size if sampler.n_steps > 0 else 0
        self.sampler = sampler if sampler.n_steps > 0 else (lambda x: [x])
        self.callbacks = callbacks if callbacks else []
        self.start_iter = start_iter
        self.eval_every = eval_every
        self.meta_rate = meta_rate if sampler.n_steps > 0 else 0.0
        self.replay_buffer = ReplayBuffer(self.replay_size, self.gan.gen.z_dim)
        self.step_every = step_every

        self.log_step_size_param = torch.nn.Parameter(
            torch.tensor(sampler.mcmc_args["step_size"]).log()
            )
        sampler.mcmc_args["step_size"] = self.log_step_size_param.exp()
        self.step_opt = torch.optim.Adam([self.log_step_size_param], lr=1e-4)

        self.sampler = sampler if sampler.n_steps > 0 else DumbSampler()

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

    @staticmethod
    def compute_grad_norm(model):
        total_norm = 0
        parameters = [
            p for p in model.parameters() if p.grad is not None and p.requires_grad
        ]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def step(self, real_batch: torch.Tensor, use_sampler: bool = True) -> Tuple[float, ...]:
        prior = self.gan.gen.prior
        d_batch = real_batch
        d_batch.requires_grad_(True)
        batch_size = d_batch.shape[0]

        self.step_opt.zero_grad()
        self.freeze_gen()
        self.unfreeze_dis()
        self.optimizer_d.zero_grad()
        loss_d_num = 0
        grad_norm_d_num = 0
        for step_id in range(1, self.n_dis * self.grad_acc_steps + 1):
            latent_start = prior.sample((batch_size,))

            if len(self.replay_buffer) > batch_size:
                #replay = np.random.rand(1) < self.replay_prob
                replay_mask = (torch.rand(batch_size) < self.replay_prob).float()[:, None]
                latent_start = (1.0 - replay_mask) * latent_start.cpu() + replay_mask * self.replay_buffer.pop(batch_size)
                #latent_start = self.replay_buffer.pop(batch_size) if replay else latent_start
                latent_start = latent_start.to(self.device)

            self.sampler.mcmc_args["step_size"] = self.log_step_size_param.exp()
            if use_sampler:
                latents, _, _, _ = self.sampler(latent_start, keep_graph=True)
            else:
                gamma = self.sampler.mcmc_args["step_size"]
                #var = latents[-1].std(dim=0)
                latents = [(1 - 2 * gamma)**.5 * latent_start + prior.sample((batch_size,)) * (2 * gamma)**.5]
                #latents = [latent_start + prior.sample((batch_size,)) * (2 * gamma)**.5]
                #latents = [1.1 * prior.sample((batch_size,))]
            drop_mask = (
                (torch.rand(batch_size) < self.meta_rate)
                .to(self.device)
                .float()[:, None]
            )
            latent = (1.0 - drop_mask) * prior.sample((batch_size,)) + drop_mask * latents[-1].to(
                self.device
            )
            self.replay_buffer.push(latents[-1])

            fake_batch = self.gan.gen(latent)
            score_fake = self.gan.dis(fake_batch).squeeze()
            score_real = self.gan.dis(d_batch).squeeze()

            loss_d = self.criterion_d(score_fake, score_real)
            if self.gp_coef > 0:
                loss_d += gradient_penalty(
                    self.gan.dis, d_batch, fake_batch, self.gp_coef
                )
            elif self.r1_coef > 0:
                loss_d += r1_penalty(
                    score_real, d_batch, self.r1_coef
                )
            loss_d /= self.grad_acc_steps
            loss_d_ = loss_d #- 1.0 * (torch.norm(latent - latent_start, dim=1)**2).mean(0)
            loss_d_.backward()
            loss_d_num += loss_d.item()
            grad_norm_d_num += (
                self.compute_grad_norm(self.gan.dis) / self.grad_acc_steps
            )
            if step_id % self.grad_acc_steps == 0:
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()
                # self.step_opt.step()
                # self.step_opt.zero_grad()

        self.step_opt.zero_grad()
        self.freeze_dis()
        self.unfreeze_gen()
        self.optimizer_g.zero_grad()
        loss_g_num = 0
        grad_norm_g_num = 0
        for step_id in range(1, self.n_gen * self.grad_acc_steps + 1):
            latent_start = prior.sample((batch_size,))

            if len(self.replay_buffer) > batch_size:
                #replay = np.random.rand(1) < self.replay_prob
                replay_mask = (torch.rand(batch_size) < self.replay_prob).float()[:, None]
                latent_start = (1.0 - replay_mask) * latent_start.cpu() + replay_mask * self.replay_buffer.pop(batch_size)
                #latent_start = self.replay_buffer.pop(batch_size) if replay else latent_start
                latent_start = latent_start.to(self.device)

            self.sampler.mcmc_args["step_size"] = self.log_step_size_param.exp()
            if use_sampler:
                latents, _, _, _ = self.sampler(latent_start, keep_graph=True)
            else:
                gamma = self.sampler.mcmc_args["step_size"]
                latents = [(1 - 2 * gamma)**.5 * latent_start + prior.sample((batch_size,)) * (2 * gamma)**.5]
                #latents = [latent_start + prior.sample((batch_size,)) * (2 * gamma)**.5]
                #latents = [1.1 * prior.sample((batch_size,))]
            drop_mask = (
                (torch.rand(batch_size) < self.meta_rate)
                .to(self.device)
                .float()[:, None]
            )
            latent = (1.0 - drop_mask) * prior.sample((batch_size,)) + drop_mask * latents[-1].to(
                self.device
            )
            self.replay_buffer.push(latents[-1])
            
            fake_batch = self.gan.gen(latent)
            score_fake = self.gan.dis(fake_batch).squeeze()

            loss_g = self.criterion_g(score_fake)
            loss_g /= self.grad_acc_steps

            loss_g_ = loss_g
            loss_g_.backward(retain_graph=True)
            loss_g_num += loss_g.item()
            grad_norm_g_num += (
                self.compute_grad_norm(self.gan.gen) / self.grad_acc_steps
            )
            if step_id % self.grad_acc_steps == 0:
                self.optimizer_g.step()
                self.optimizer_g.zero_grad()
                # self.step_opt.step()
                # self.step_opt.zero_grad()

        return loss_g_num, loss_d_num, grad_norm_d_num, grad_norm_g_num

    def train(self):
        self.gan.dis.train()
        self.gan.gen.train()
        loss_g, loss_d = [], []
        for batch_id, batch in tqdm(
            enumerate(self.dataloader, 1), total=len(self.dataloader)
        ):
            if batch_id < self.start_iter:
                continue
            # if (batch_id - 1) % 1000 == 0 and not isinstance(self.sampler, DumbSampler):
            #     self.sampler.feature.reset()
            batch = batch.to(self.device)
            use_sampler = (batch_id % self.step_every == 1) and self.step_every > 0
            l_g, l_d, grad_norm_d, grad_norm_g = self.step(batch, use_sampler=use_sampler)
            loss_g = loss_g[-self.eval_every + 1 :] + [l_g]
            loss_d = loss_d[-self.eval_every + 1 :] + [l_d]

            info = dict(
                step=batch_id,
                total=len(self.dataloader),
                loss_g=np.mean(loss_g),
                loss_d=np.mean(loss_d),
                grad_norm_d=grad_norm_d,
                grad_norm_g=grad_norm_g,
            )

            if batch_id % self.eval_every == 0:
                self.gan.gen.eval()
                imgs = []
                sample_size = self.sample_size
                if len(self.replay_buffer) > 0:
                    latent = self.replay_buffer.pop(batch.shape[0])
                while sample_size > 0:
                    if not isinstance(self.sampler, DumbSampler):
                        latents, _, _, _ = self.sampler(latent, keep_graph=False)
                        latent = latents[-1].to(self.device)
                    else:
                        latent = self.gan.gen.prior.sample((batch.shape[0],))  
                    with torch.no_grad():
                        fake_batch = self.gan.gen(latent[:sample_size])
                        
                    imgs.append(
                        self.gan.gen.inverse_transform(fake_batch)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    sample_size -= batch.shape[0]
                imgs = np.concatenate(imgs, axis=0)

                if not isinstance(self.sampler, DumbSampler):
                    weight = self.sampler.feature.weight
                    weight_norm = weight[0].norm().item() if len(weight) > 0 else 0
                else:
                    weight_norm = 0
                
                info.update(
                    imgs=imgs,
                    weight_norm=weight_norm,
                )
                self.gan.gen.train()

                if not isinstance(self.sampler, DumbSampler):
                    print(f'step size: {self.sampler.mcmc_args["step_size"]}')
                    print(f'{self.sampler.feature.weight}')

            for callback in self.callbacks:
                callback.invoke(info)
