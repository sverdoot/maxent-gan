from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import torch


class Distribution(ABC):
    """Abstract class for distribution"""

    @abstractmethod
    def log_prob(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Computes log probability of input z"""
        raise NotImplementedError


class DistributionRegistry:
    registry: Dict = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        def inner_wrapper(wrapped_class: Distribution) -> Callable:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Distribution:
        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor


@DistributionRegistry.register()
class PriorTarget(Distribution):
    def __init__(
        self,
        gan,
    ):
        self.gan = gan
        self.proposal = gan.prior

    def log_prob(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.proposal.log_prob(z)

    def project(self, z):
        return self.proposal.project(z)


@DistributionRegistry.register()
class DiscriminatorTarget(Distribution):
    def __init__(self, gan, batch_size=None):
        self.gan = gan
        self.proposal = gan.prior
        self.batch_size = batch_size

    def log_prob(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        batch_size = kwargs.get("batch_size", self.batch_size or len(z))
        log_prob = torch.empty((0,), device=z.device)
        for chunk in torch.split(z, batch_size):
            dgz = self.gan.dis(self.gan.gen(chunk)).squeeze()
            logp_z = self.proposal.log_prob(chunk)
            if dgz.shape != logp_z.shape:
                raise Exception
            log_prob = torch.cat([log_prob, (logp_z + dgz) / 1.0])
        return log_prob

    def project(self, z):
        return self.proposal.project(z)


@DistributionRegistry.register()
class CondTarget(Distribution):
    def __init__(
        self,
        gan,
        data_batch: Optional[torch.FloatTensor] = None,
        batch_size=None,
    ):
        self.gan = gan
        self.proposal = gan.prior
        self.data_batch = data_batch
        self.batch_size = batch_size

    def log_prob(
        self,
        z: torch.FloatTensor,
        data_batch: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size = kwargs.get("batch_size", self.batch_size or len(z))
        data_batch = data_batch if data_batch is not None else self.data_batch
        log_prob = torch.empty((0,), device=z.device)
        for chunk, data_chunk in zip(
            torch.split(z, batch_size), torch.split(data_batch, batch_size)
        ):
            logp_xz = (
                -torch.norm(
                    (self.gan.gen(chunk) - data_chunk.to(chunk.device)).reshape(
                        len(chunk), -1
                    ),
                    dim=1,
                )
                ** 2
                / 2
            )
            logp_z = self.proposal.log_prob(chunk)
            if logp_xz.shape != logp_z.shape:
                raise Exception
            log_prob = torch.cat([log_prob, (logp_z + logp_xz) / 1.0])

        return log_prob

    def project(self, z):
        return self.proposal.project(z)


@DistributionRegistry.register()
class MaxEntTarget(Distribution):
    def __init__(self, gen, feature, ref_dist, batch_size=None):
        self.gen = gen
        self.feature = feature
        self.ref_dist = ref_dist
        self.proposal = gen.prior
        self.batch_size = batch_size
        self.radnic_logps = []
        self.ref_logps = []

    def log_prob(
        self,
        z: torch.FloatTensor,
        data_batch: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size = kwargs.get("batch_size", self.batch_size or len(z))
        log_prob = torch.empty((0,), device=z.device)

        for chunk in torch.split(z, batch_size):
            f = self.feature(self.gen(chunk), chunk)
            radnic_logp = self.feature.log_prob(f)
            ref_logp = self.ref_dist.log_prob(chunk, data_batch=data_batch)
            if not isinstance(radnic_logp, torch.Tensor):
                radnic_logp = torch.zeros_like(ref_logp, device=ref_logp.device)
            log_prob = torch.cat([log_prob, radnic_logp + ref_logp])

            self.radnic_logps.append(radnic_logp.detach())
            self.ref_logps.append(ref_logp.detach())

        return log_prob

    def project(self, z):
        return self.proposal.project(z)


def grad_log_prob(
    point: torch.FloatTensor,
    log_dens: Union[Distribution, Callable],
    # x: Optional[Any] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    point = point.detach().requires_grad_()
    log_prob = log_dens(point)
    grad = torch.autograd.grad(log_prob.sum(), point)[0]
    return log_prob, grad


# def estimate_log_norm_constant(
#     gen: nn.Module,
#     dis: nn.Module,
#     n_pts: int,
#     batch_size: Optional[int] = None,
#     verbose: bool = False,
# ) -> float:
#     batch_size = batch_size if batch_size else n_pts
#     norm_const = 0
#     if verbose:
#         bar = trange
#     else:
#         bar = range

#     for _ in bar(0, n_pts, batch_size):
#         z = gen.prior.sample((batch_size,))
#         label = gen.sample_label(batch_size, z.device)
#         gen.label = label
#         dis.label = label

#         dgz = dis(gen(z)).squeeze()
#         norm_const += (dgz).exp().sum().item()
#     norm_const /= n_pts

#     log_norm_const = float(np.log(norm_const))
#     return log_norm_const


# def harmonic_mean_estimate(
#     dis: nn.Module,
#     x: Union[np.ndarray, torch.FloatTensor],
#     label: Union[np.ndarray, torch.FloatTensor],
#     device,
#     batch_size: Optional[int] = None,
#     verbose: bool = False,
# ) -> float:
#     batch_size = batch_size if batch_size else len(x)
#     inv_norm_const = 0
#     if isinstance(x, np.ndarray):
#         x = torch.from_numpy(x)
#     if isinstance(label, np.ndarray):
#         label = torch.from_numpy(label)

#     for i, x_batch in enumerate(torch.split(x, batch_size)):
#         label_batch = label[i * batch_size : (i + 1) * batch_size]
#         dis.label = label_batch.to(device)

#         dgz = dis(x_batch.to(device)).squeeze()
#         inv_norm_const += (-dgz).exp().sum().item()
#     inv_norm_const /= len(x)

#     log_norm_const = float(-np.log(inv_norm_const))
#     return log_norm_const
