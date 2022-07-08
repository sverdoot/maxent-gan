from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch.autograd import grad
from torch.nn import functional as F


class Loss(ABC):
    @abstractmethod
    def __call__(
        self, fake_score: torch.Tensor, real_score: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError


class LossRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        def inner_wrapper(wrapped_class: Loss) -> Loss:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Loss:
        model = cls.registry[name]
        model = model(**kwargs)
        return model


@LossRegistry.register()
class JensenLoss(Loss):
    @classmethod
    def __call__(
        cls,
        fake_score: torch.Tensor,
        real_score: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = fake_score.device

        if real_score is not None:
            ones = torch.ones_like(real_score).to(device)
            loss = F.binary_cross_entropy_with_logits(
                real_score, ones, reduction="mean", weight=weight
            )
            zeros = torch.zeros_like(fake_score).to(device)
            loss += F.binary_cross_entropy_with_logits(fake_score, zeros, weight=weight)
            return loss
        else:
            # ones = torch.ones_like(fake_score).to(device)
            zeros = torch.zeros_like(fake_score).to(device)
            loss = -F.binary_cross_entropy_with_logits(
                fake_score, zeros, reduction="mean", weight=weight
            )
            return loss


@LossRegistry.register()
class JensenNSLoss(Loss):
    @classmethod
    def __call__(
        cls,
        fake_score: torch.Tensor,
        real_score: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = fake_score.device
        if weight is None:
            weight = torch.ones_like(fake_score).to(fake_score.device)

        if real_score is not None:
            ones = torch.ones_like(real_score).to(device)
            loss = F.binary_cross_entropy_with_logits(
                real_score, ones, reduction="mean"  # , weight=weight
            )
            # print(real_score.mean(), fake_score.mean())
            zeros = torch.zeros_like(fake_score).to(device)
            loss += F.binary_cross_entropy_with_logits(fake_score, zeros, weight=weight)
            return loss
        else:
            # ones = torch.ones_like(fake_score).to(device)
            ones = torch.ones_like(fake_score).to(device)
            loss = F.binary_cross_entropy_with_logits(
                fake_score, ones, reduction="mean", weight=weight
            )
            return loss


@LossRegistry.register()
class Wasserstein1Loss(Loss):
    @classmethod
    def __call__(
        cls,
        fake_score: torch.Tensor,
        real_score: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if weight is None:
            weight = torch.ones_like(fake_score)
        if real_score is not None:
            loss = (fake_score * weight.detach()).mean()
            loss -= real_score.mean()
            return loss
        else:
            loss = -(fake_score * weight.detach()).mean()
            return loss


@LossRegistry.register()
class HingeLoss(Loss):
    @classmethod
    def __call__(
        cls, fake_score: torch.Tensor, real_score: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if real_score is None:
            loss = -fake_score.mean()
            return loss
        else:
            loss = (
                -torch.clip(-1 + real_score, min=0).mean()
                - torch.clip(-1 - fake_score, min=0).mean()
            )
            return loss


def gradient_penalty(
    dis,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    gp_coef: float,
):
    alpha = torch.rand(real_data.shape[0], *[1] * (real_data.ndim - 1))
    alpha = alpha.expand(real_data.size()).to(real_data.device)

    mix = alpha * real_data + (1 - alpha) * fake_data
    mix.requires_grad_(True)
    dis_mix = dis(mix).squeeze()
    ones = torch.ones_like(dis_mix, device=real_data.device)
    grads = grad(
        outputs=dis_mix,
        inputs=mix,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_penalty = ((grads.norm(p=2, dim=1) - 1.0) ** 2).mean() * gp_coef
    return grad_penalty
