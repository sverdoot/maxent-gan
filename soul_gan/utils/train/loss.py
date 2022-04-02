import torch
from torch.nn import functional as F
from typing import Optional, Callable
from abc import ABC, abstractmethod


class Loss(ABC):

    @abstractmethod
    def __call__(self, 
        fake_score: torch.Tensor, 
        real_score: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    def __call__(cls, fake_score: torch.Tensor, real_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = fake_score.device

        if real_score is not None:
            ones = torch.ones_like(real_score).to(device)
            loss = F.binary_cross_entropy_with_logits(real_score, ones, reduction='mean')
            zeros = torch.zeros_like(fake_score).to(device)
            loss += F.binary_cross_entropy_with_logits(fake_score, zeros)
            return loss
        else:
            ones = torch.ones_like(fake_score).to(device)
            loss = F.binary_cross_entropy_with_logits(fake_score, ones, reduction='mean')
            return loss


@LossRegistry.register()
class Wass1Loss(Loss):
    @classmethod
    def __call__(cls, fake_score: torch.Tensor, real_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        if real_score is not None:
            loss = fake_score.mean()
            loss -= real_score.mean()
            return loss
        else:
            loss = -fake_score.mean()
            return loss


@LossRegistry.register()
class HingeLoss(Loss):
    @classmethod
    def __call__(cls, fake_score: torch.Tensor, real_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        if real_score is None:
            loss = -fake_score.mean()
            return loss
        else:
            loss = - torch.clip(-1 + real_score, min=0).mean() - \
                torch.clip(-1 - fake_score, min=0).mean()
            return loss
        