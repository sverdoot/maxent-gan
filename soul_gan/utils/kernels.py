from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch


class Kernel(ABC):
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    @abstractmethod
    def __call__(self, x, y):
        raise NotImplementedError

    def unsqueeze_bandwidth(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if y is not None and x.ndim != y.ndim:
            raise Exception
        if isinstance(self.bandwidth, torch.Tensor) and self.bandwidth.ndim == 1:
            if x.ndim == 2:
                bandwidth = self.bandwidth[None, :].to(x.device)
            elif x.ndim == 3:
                bandwidth = self.bandwidth[None, None, :].to(x.device)
        elif isinstance(self.bandwidth, torch.Tensor) and self.bandwidth.ndim == 0:
            bandwidth = self.bandwidth.item()
        else:
            bandwidth = self.bandwidth
        return bandwidth


class KernelRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        def inner_wrapper(wrapped_class: Kernel) -> Kernel:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Kernel:
        kernel = cls.registry[name]
        kernel = kernel(**kwargs)
        return kernel


@KernelRegistry.register()
class LinearKernel(Kernel):
    def __call__(self, x, y):
        bandwidth = self.unsqueeze_bandwidth(x, y)
        return (x * y / bandwidth ** 2).sum(dim=-1)


@KernelRegistry.register()
class PolynomialKernel(Kernel):
    def __init__(self, bandwidth=1.0, d=2):
        super().__init__(bandwidth)
        self.d = d

    def __call__(self, x, y):
        bandwidth = self.unsqueeze_bandwidth(x, y)
        return ((x * y / bandwidth ** 2).sum(dim=-1) + 1) ** self.d


@KernelRegistry.register()
class GaussianKernel(Kernel):
    def __call__(self, x, y):
        bandwidth = self.unsqueeze_bandwidth(x, y)
        return torch.exp(
            -((torch.norm((x - y) / (2 * bandwidth), dim=-1, p=2)) ** 2)
        )
