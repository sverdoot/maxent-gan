from typing import Optional

from torch import nn


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
