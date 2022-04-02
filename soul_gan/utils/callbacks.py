from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from matplotlib import pyplot as plt

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid

import ot


class Callback(ABC):
    cnt: int = 0

    @abstractmethod
    def invoke(self):
        pass

    def reset(self):
        self.cnt = 0


class CallbackRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        def inner_wrapper(wrapped_class: Callback) -> Callback:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Callback:
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
        self.init_params = init_params if init_params else {}
        import wandb

        self.wandb = wandb
        wandb.init(**self.init_params)

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
                                torch.clip(torch.from_numpy(info[key][:25]), 0, 1)
                            ),
                            nrow=5,
                        ),
                        caption=key,
                    )
                else:
                    log[key] = info[key]
            log["step"] = self.cnt if "step" not in info else info["step"]
            wandb.log(log)
        self.cnt += 1
        return 1

    def reset(self):
        super().reset()
        self.wandb.init(**self.init_params)


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


@CallbackRegistry.register()
class DiscriminatorCallback(Callback):
    def __init__(
        self,
        gan,
        invoke_every=1,
        update_input=True,
        device="cuda",
        batch_size: Optional[int] = None,
    ):
        self.invoke_every = invoke_every
        self.dis = gan.dis
        self.transform = self.dis.transform
        self.update_input = update_input
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def invoke(
        self,
        info: Dict[str, Union[float, np.ndarray]],
        batch_size: Optional[int] = None,
    ):
        dgz = None
        if self.cnt % self.invoke_every == 0:
            imgs = info["imgs"]
            if "label" in info:
                label = torch.LongTensor(info["label"]).to(self.device)
            else:
                label = None

            if not batch_size:
                batch_size = len(imgs) if not self.batch_size else self.batch_size
            x = self.transform(torch.from_numpy(imgs).to(self.device))
            dgz = 0
            for i, x_batch in enumerate(torch.split(x, batch_size)):
                if label is not None:
                    label_batch = label[i * batch_size : (i + 1) * batch_size]
                else:
                    label_batch = None
                self.dis.label = label_batch
                # dgz += self.dis.output_layer(self.dis(x_batch)).sum().item()
                dgz += (self.dis(x_batch).squeeze()).sum().item()
            dgz /= len(imgs)

            if self.update_input:
                info["D(G(z))"] = dgz
        self.cnt += 1

        return dgz


@CallbackRegistry.register()
class EnergyCallback(Callback):
    def __init__(
        self,
        gan,
        norm_constant=1,
        invoke_every=1,
        update_input=True,
        device="cuda",
        batch_size: Optional[int] = None,
        log_norm_const: float = 0,
    ):
        self.invoke_every = invoke_every
        self.dis = gan.dis
        self.gen = gan.gen
        self.norm_constant = norm_constant
        self.transform = self.dis.transform
        self.update_input = update_input
        self.device = device
        self.batch_size = batch_size
        self.log_norm_const = log_norm_const

    @torch.no_grad()
    def invoke(
        self,
        info: Dict[str, Union[float, np.ndarray]],
        batch_size: Optional[int] = None,
    ):
        energy = None
        if self.cnt % self.invoke_every == 0:
            zs = torch.FloatTensor(info["zs"]).to(self.device)
            if "label" in info:
                label = torch.LongTensor(info["label"]).to(self.device)
            else:
                label = None

            if not batch_size:
                batch_size = len(zs) if not self.batch_size else self.batch_size
            energy = 0
            # log_norm_const = estimate_log_norm_constant(
            #     self.gen, self.dis, 5000
            # )

            for i, z_batch in enumerate(torch.split(zs, batch_size)):
                if label is not None:
                    label_batch = label[i * batch_size : (i + 1) * batch_size]
                    self.dis.label = label_batch
                    self.gen.label = label_batch
                else:
                    label_batch = None

                dgz = self.dis(self.gen(z_batch)).squeeze()
                energy += -(self.gen.prior.log_prob(z_batch).sum() + dgz.sum()).item()
            energy /= len(zs)
            energy += self.log_norm_const

            if self.update_input:
                info["Energy"] = energy
        self.cnt += 1

        return energy


@CallbackRegistry.register()
class LogCallback(Callback):
    def __init__(
        self,
        save_dir: Union[Path, str],
        keys: List[str],
        invoke_every: int = 1,
    ):
        self.save_dir = Path(save_dir)
        self.invoke_every = invoke_every
        self.keys = keys

        self.save_paths = []
        for key in keys:
            path = Path(save_dir, f"{key}.txt")
            path.open("w")
            self.save_paths.append(path)

    @torch.no_grad()
    def invoke(
        self,
        info: Dict[str, Union[float, np.ndarray]],
    ):
        if self.cnt % self.invoke_every == 0:
            for save_path, key in zip(self.save_paths, self.keys):
                if key not in info:
                    continue
                with save_path.open("ab") as f:
                    np.savetxt(f, [info[key]], delimiter=" ", newline=" ")
        self.cnt += 1

    def reset(self):
        super().reset()
        for save_path in self.save_paths:
            save_path.open("ab").write(b"\n")


@CallbackRegistry.register()
class EMDCallback(Callback):
    def __init__(
        self,
        np_dataset,
        invoke_every=1,
        update_input=True,
    ):
        self.invoke_every = invoke_every
        self.update_input = update_input
        self.np_dataset = np_dataset

    @torch.no_grad()
    def invoke(
        self,
        info: Dict[str, Union[float, np.ndarray]],
        batch_size: Optional[int] = None,
    ):
        emd = None
        if self.cnt % self.invoke_every == 0:
            M = ot.dist(info['imgs'], self.np_dataset)
            emd2 = ot.emd2(np.ones(info['imgs'].shape[0]) / info['imgs'].shape[0], np.ones(self.np_dataset.shape[0]) / self.np_dataset.shape[0], M)
            emd = emd2 ** .5
            if self.update_input:
                info["EMD"] = emd
        else:
            emd = 0
        self.cnt += 1

        return emd


@CallbackRegistry.register()
class Plot2dCallback(Callback):
    def __init__(
        self,
        modes: np.ndarray,
        save_dir: Union[str, Path],
        every: int, 
        invoke_every: int = 1) -> None:
        self.save_dir = Path(save_dir, 'figs')
        self.save_dir.mkdir(exist_ok=True)
        self.modes = modes
        self.invoke_every = invoke_every
        self.every = every

    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        step = self.cnt # if "step" not in info else info["step"]
        if self.cnt % self.invoke_every == 0:
            xs = info['imgs']
            plt.figure(figsize=(4, 4))
            plt.scatter(xs[:, 0], xs[:, 1], alpha=0.1, s=10)

            plt.scatter(self.modes[:, 0], self.modes[:, 1], color='r', marker='x')
            plt.axis('equal')
            # plt.grid()

            savepath = Path(self.save_dir, f'{self.save_dir.parts[-2]}_2d_{self.cnt * self.every}').as_posix()
            plt.savefig(savepath + '.png')
            plt.savefig(savepath + '.pdf')
            plt.close()
        self.cnt += 1

        return 1


@CallbackRegistry.register()
class TrainLogCallback(Callback):
    def __init__(
        self,
        invoke_every: int = 1) -> None:
        self.invoke_every = invoke_every

    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        step = self.cnt # if "step" not in info else info["step"]
        if self.cnt % self.invoke_every == 0:
            ep = info["step"]
            loss_g = info["loss_g"]
            loss_d = info["loss_d"]
            print(f'Epoch: {ep}, Loss G: {loss_g}, Loss D: {loss_d}')

        self.cnt += 1
        return 1