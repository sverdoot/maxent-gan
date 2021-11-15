from torch import nn

class ModuleRegisry:
    registry = {}

    @classmethod
    def register(cls, name: str) -> nn.Module:
        def inner_wrapper(wrapped_class: nn.Module) -> nn.Module:

            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_model(cls, name: str, **kwargs) -> nn.Module:
        model = cls.registry[name]
        model = model(**kwargs)
        return model