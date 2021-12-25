from typing import Optional, Tuple

import torch
from torch import nn
from torchvision import transforms


class ModelRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> nn.Module:
        def inner_wrapper(wrapped_class: nn.Module) -> nn.Module:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_model(cls, name: str, **kwargs) -> nn.Module:
        model = cls.registry[name]
        model = model(**kwargs)
        return model


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-10)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class BaseDiscriminator(nn.Module):
    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        output_layer: str,
    ):
        super().__init__()
        self.transform = transforms.Normalize(mean, std)
        if output_layer == "sigmoid":
            self.output_layer = nn.Sigmoid()
        else:
            self.output_layer = nn.Identity()


class BaseGenerator(nn.Module):
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ):
        super().__init__()
        self.inverse_transform = NormalizeInverse(mean, std)
