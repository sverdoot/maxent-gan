from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid


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
        self,
        invoke_every: int = 1,
        init_params: Optional[Dict] = None,
        keys: Optional[List[str]] = None,
    ):
        init_params = init_params if init_params else {}
        import wandb

        self.wandb = wandb
        wandb.init(**init_params)

        self.invoke_every = invoke_every
        self.keys = keys

        self.img_transform = transforms.Resize(
            128, interpolation=transforms.InterpolationMode.NEAREST
        )

    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        if self.cnt % self.invoke_every == 0:
            wandb = self.wandb
            if not self.keys:
                self.keys = info.keys()
            log = dict()
            for key in self.keys:
                if key not in info:
                    continue
                if isinstance(info[key], np.ndarray):
                    log[key] = wandb.Image(
                        make_grid(
                            self.img_transform(
                                torch.from_numpy(info[key][:25])
                            ),
                            nrow=5,
                        ),
                        caption=key,
                    )
                else:
                    log[key] = info[key]
            log["step"] = self.cnt
            wandb.log(log)
        self.cnt += 1


@CallbackRegistry.register()
class SaveImagesCallback(Callback):
    def __init__(
        self,
        save_dir: Union[Path, str],
        invoke_every: int = 1,
    ):
        self.invoke_every = invoke_every
        self.save_dir = save_dir

    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        if self.cnt % self.invoke_every == 0:
            imgs = info["imgs"]
            savepath = Path(self.save_dir, f"imgs_{self.cnt}.npy")
            np.save(savepath, imgs)

        self.cnt += 1


# class SaveImagesCallback(Callback):
#     pass


@CallbackRegistry.register()
class DiscriminatorCallback(Callback):
    def __init__(self, dis, invoke_every=1, update_input=True, device="cuda"):
        self.invoke_every = invoke_every
        self.dis = dis
        self.transform = dis.transform
        self.update_input = update_input
        self.device = device

    @torch.no_grad()
    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        dgz = None
        if self.cnt % self.invoke_every == 0:
            imgs = info["imgs"]
            x = self.transform(torch.from_numpy(imgs).to(self.device))
            dgz = torch.sigmoid(self.dis(x)).mean().item()
            if self.update_input:
                info["D(G(z))"] = dgz
        self.cnt += 1

        return dgz
