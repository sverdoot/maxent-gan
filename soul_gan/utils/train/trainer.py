from typing import Callable, Optional, Iterable, Tuple

from soul_gan.utils.callbacks import Callback


class Trainer:
    def __init__(
            self, 
            gan, 
            optimizer_g, 
            optimizer_d, 
            criterion_g, 
            criterion_d,
            dataloader,
            device,
            n_dis=3,
            sample_fn: Optional[Callable] = None,
            callbacks: Optional[Iterable[Callback]]=None
            ):
        self.gan = gan
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.criterion_g = criterion_g
        self.criterion_d = criterion_d
        self.dataloader = dataloader
        self.device = device
        self.n_dis = n_dis
        self.sample_fn = sample_fn
        self.callbacks = callbacks if callbacks else []

    def sample_fake(self, batch_size: int, sample_fn: Optional[Callable]=None):
        latent = self.gan.gen.prior.sample((batch_size,))
        if sample_fn:
            latents, fakes = sample_fn(latent)
            fake_batch = self.gan.gen(latents[-1].to(self.device))
        else:
            fake_batch = self.gan.gen(latent)
        return fake_batch

    def step(self, real_batch) -> Tuple[float, float]:
        batch_size = real_batch.shape[0]

        for _ in range(self.n_dis):
            fake_batch = self.sample_fake(batch_size, sample_fn=None)
            score_real = self.gan.dis(real_batch)
            score_fake = self.gan.dis(fake_batch)
            loss_d = self.criterion_d(score_fake, score_real)
            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

        fake_batch = self.sample_fake(batch_size, sample_fn=self.sample_fn)
        score_fake = self.gan.dis(fake_batch)
        loss_g = self.criterion_g(score_fake)
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.item(), loss_d.item()

    def train(self, n_epochs, report_every=10):
        for ep in range(n_epochs):
            loss_g, loss_d = 0, 0
            for batch in self.dataloader:
                batch = batch.to(self.device)
                l_g, l_d = self.step(batch)
                loss_g += l_g / len(self.dataloader)
                loss_d += l_d

            imgs = self.gan.gen.inverse_transform(self.gan.gen(
                self.gan.gen.prior.sample((batch.shape[0],))
            )).detach().cpu().numpy()
            info = dict(
                step=ep, 
                loss_g=loss_g, 
                loss_d=loss_d,
                imgs=imgs
            )

            for callback in self.callbacks:
                callback.invoke(info)

