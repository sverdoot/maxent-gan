from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

from soul_gan.utils.metrics.inception_score import batch_inception


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
            wandb.log(log)
        self.cnt += 1


@CallbackRegistry.register()
class InceptionScoreCallback(Callback):
    def __init__(
        self,
        invoke_every: int = 1,
        device: Union[str, int, torch.device] = "cuda",
        update_input=True,
    ):
        self.device = device
        self.model = torchvision.models.inception.inception_v3(
            pretrained=True, transform_input=False
        ).to(device)
        self.model.eval()
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.update_input = update_input
        self.invoke_every = invoke_every

    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        score = None
        if self.cnt % self.invoke_every == 0:
            imgs = torch.from_numpy(info["imgs"]).to(self.device)
            imgs = self.transform(imgs)
            pis = batch_inception(imgs, self.model, resize=True)
            score = (
                (pis * (torch.log(pis) - torch.log(pis.mean(0)[None, :])))
                .sum(1)
                .mean(0)
            )
            score = torch.exp(score)

            if self.update_input:
                info["inception score"] = score
        self.cnt += 1
        return score


class SaveLatentsCallback(Callback):
    pass


class FIDCallback(Callback):
    pass


class SaveImagesCallback(Callback):
    pass
