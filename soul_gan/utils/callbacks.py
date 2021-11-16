from abc import ABC, abstractmethod
from typing import Dict, Optional


class Callback(ABC):
    cnt: int = 0

    @abstractmethod
    def invoke(self):
        pass


class CallbackRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callback:
        def inner_wrapper(wrapped_class: Callback) -> Callback:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_callback(cls, name: str, **kwargs) -> Callback:
        model = cls.registry[name]
        model = model(**kwargs)
        return model


@CallbackRegistry.register()
class WandbCallback(Callback):
    def __init__(
        self, invoke_every: int = 1, init_params: Optional[Dict] = None
    ):
        init_params = init_params if init_params else {}
        import wandb

        self.wandb = wandb
        wandb.init(**init_params)

        self.invoke_every = invoke_every

    def invoke(self, info: Dict[str, float]):
        if self.cnt % self.invoke_every == 0:
            wandb = self.wandb
            wandb.log(info)
        self.cnt += 1
