from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torchvision
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms

from soul_gan.utils.metrics import batch_inception

# from main.nnclass import CNN
# from main.util.save_util import save_image_general, save_weight_vgg_feature


class AvgHolder(object):
    cnt: int = 0

    def __init__(self, init_val: Any = 0):
        self.val = init_val

    def upd(self, new_val: Any):
        self.cnt += 1
        alpha = 1.0 / self.cnt
        if isinstance(self.val, list):
            for i in range(len(self.val)):
                self.val[i] = self.val[i] * (1.0 - alpha) + new_val[i] * alpha
        else:
            self.val = self.val * (1.0 - alpha) + new_val * alpha

    def reset(self):
        if isinstance(self.val, list):
            self.val = [0] * len(self.val)
        else:
            self.val = 0
        self.cnt = 0

    @property
    def data(self) -> Any:
        return self.val


class Feature(ABC):
    def __init__(
        self,
        n_features: int = 1,
        callbacks: Optional[List] = None,
        inverse_transform=None,
        device="cuda",
    ):
        self.n_features = n_features
        self.device = device
        self.avg_weight = AvgHolder([0] * n_features)
        self.avg_feature = AvgHolder([0] * n_features)

        self.callbacks = callbacks if callbacks else []

        if not inverse_transform:
            self.inverse_transform = (
                None  # transforms.Normalize((0, 0, 0), (1, 1, 1))
            )
        else:
            self.inverse_transform = inverse_transform

    def log_prob(self, out: List[torch.FloatTensor]) -> torch.FloatTensor:
        lik_f = 0
        for feature_id in range(len(out)):
            # lik_f -= torch.dot(self.weight[feature_id], out[feature_id])
            lik_f -= out[feature_id] @ self.weight[feature_id]

        return lik_f

    def weight_up(self, out: List[torch.FloatTensor], step: float):
        for i in range(len(self.weight)):
            if isinstance(out[i], torch.FloatTensor):
                grad = out[i].mean(0)
            else:
                grad = out[i]
            self.weight[i] += step * grad
            self.project_weight()

    @staticmethod
    def average_feature(feature_method: Callable) -> Callable:
        # @wraps
        def with_avg(self, *args, **kwargs):
            out = feature_method(self, *args, **kwargs)
            self.avg_feature.upd([x.mean(0) for x in out])
            return out

        return with_avg

    @staticmethod
    def get_useful_info(
        x: torch.FloatTensor, feature_out: List[torch.FloatTensor]
    ) -> Dict:
        return {
            f"feature_{i}": val.mean().item()
            for i, val in enumerate(feature_out)
        }

    @staticmethod
    def invoke_callbacks(feature_method: Callable) -> Callable:
        # @wraps
        def with_callbacks(self, x, *args, **kwargs):
            out = feature_method(self, x, *args, **kwargs)
            if self.callbacks:
                info = self.get_useful_info(x, out)
                for callback in self.callbacks:
                    callback.invoke(info)
            return out

        return with_callbacks

    @abstractmethod
    def __call__(self, x: torch.FloatTensor):
        raise NotImplementedError

    def project_weight(self):
        for i in range(len(self.weight)):
            self.weight[i] = torch.clip(self.weight[i], -1e4, 1e4)


class FeatureRegistry:
    registry: Dict = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        def inner_wrapper(wrapped_class: Feature) -> Callable:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_feature(cls, name: str, **kwargs) -> Feature:
        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor


@FeatureRegistry.register("inception_score")
class InceptionScoreFeature(Feature):
    def __init__(
        self,
        callbacks: Optional[List] = None,
        inverse_transform=None,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        **kwargs,
    ):
        super().__init__(
            n_features=1,
            callbacks=callbacks,
            inverse_transform=inverse_transform,
        )
        self.device = kwargs.get("device", 0)
        self.model = torchvision.models.inception.inception_v3(
            pretrained=True, transform_input=False
        ).to(self.device)
        self.model.eval()
        self.transform = transforms.Normalize(mean, std)

        self.ref_feature = kwargs.get("ref_score", [np.log(9.0)])
        self.weight = [torch.zeros(1).to(self.device)]

        self.pis_mean = None  # torch.zeros(1000).to(self.device)
        self.exp_avg_coef = 0.1

    # @staticmethod
    def get_useful_info(
        self, x: torch.FloatTensor, feature_out: List[torch.FloatTensor]
    ) -> Dict:
        return {
            "inception score": np.exp(
                feature_out[0].mean().item() + self.ref_feature[0]
            ),
            f"weight_{self.__class__.__name__}": torch.norm(
                self.weight[0]
            ).item(),
            "imgs": self.inverse_transform(x).detach().cpu().numpy(),
        }

    @Feature.average_feature
    @Feature.invoke_callbacks
    def __call__(self, x) -> List[torch.FloatTensor]:
        x = self.inverse_transform(x)
        x = self.transform(x)
        pis = batch_inception(x, self.model, resize=True)

        if not self.pis_mean:
            self.pis_mean = pis.mean(0).detach()
        else:
            self.pis_mean = (
                1.0 - self.exp_avg_coef
            ) * self.pis_mean + self.exp_avg_coef * pis.mean(0).detach()
        score = (
            (pis * (torch.log(pis) - torch.log(self.pis_mean[None, :])))
            .sum(1)
            .reshape(-1, 1)
        )
        score -= self.ref_feature[0]
        return [score]


@FeatureRegistry.register()
class DiscriminatorFeature(Feature):
    def __init__(self, dis, inverse_transform=None, callbacks=None, **kwargs):
        super().__init__(
            n_features=1,
            inverse_transform=inverse_transform,
            callbacks=callbacks,
        )
        self.device = kwargs.get("device", 0)
        self.dis = dis
        self.ref_feature = kwargs.get("ref_score", [np.log(0.5 / (1 - 0.5))])

        self.weight = [torch.zeros(1).to(self.device)]

    def get_useful_info(
        self, x: torch.FloatTensor, feature_out: List[torch.FloatTensor]
    ) -> Dict:
        return {
            "D(G(z))": torch.mean(
                torch.sigmoid(feature_out[0] + self.ref_feature[0])
            ).item(),
            f"weight_{self.__class__.__name__}": torch.norm(
                self.weight[0]
            ).item(),
            "imgs": self.inverse_transform(x).detach().cpu().numpy(),
        }

    @Feature.average_feature
    @Feature.invoke_callbacks
    def __call__(self, x) -> List[torch.FloatTensor]:
        score = self.dis(x).reshape(-1, 1)
        score -= self.ref_feature[0]

        return [score]


@FeatureRegistry.register()
class InceptionV3MeanFeature(Feature):
    IDX_TO_DIM = {0: 64, 1: 192, 2: 768, 3: 2048}

    def __init__(
        self, inverse_transform=None, callbacks=None, dp=False, **kwargs
    ):
        super().__init__(
            n_features=1,
            inverse_transform=inverse_transform,
            callbacks=callbacks,
        )
        self.device = kwargs.get("device", 0)
        self.block_ids = kwargs.get("block_ids", [3])
        self.data_stat_path = kwargs.get("data_stat_path")

        mean = torch.from_numpy(np.load(Path(self.data_stat_path))["mu"]).to(
            self.device
        )

        self.model = InceptionV3(self.block_ids).to(self.device)
        if dp:
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

        feature_dims = [self.IDX_TO_DIM[idx] for idx in self.block_ids]

        self.ref_feature = [
            torch.zeros(dim, device=self.device) for dim in feature_dims
        ]

        self.ref_feature = [mean]

        self.weight = [
            torch.zeros(dim, device=self.device) for dim in feature_dims
        ]

    def get_useful_info(
        self, x: torch.FloatTensor, feature_out: List[torch.FloatTensor]
    ) -> Dict:
        return {
            "feature": feature_out[0].mean().item()
            + self.ref_feature[0].mean().item(),  # noqa: W503
            f"weight_{self.__class__.__name__}": torch.norm(
                self.weight[0]
            ).item(),
            "imgs": self.inverse_transform(x).detach().cpu().numpy(),
        }

    @Feature.invoke_callbacks
    @Feature.average_feature
    def __call__(self, x) -> List[torch.FloatTensor]:
        x = self.inverse_transform(x)
        pred = self.model(x)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        out = [pred.squeeze(3).squeeze(2)]

        for i in range(len(out)):
            out[i] = (out[i] - self.ref_feature[i][None, :]).float()

        return out


@FeatureRegistry.register()
class DumbFeature(Feature):
    def __init__(
        self,
        n_features: int = 1,
        callbacks: Optional[List] = None,
        inverse_transform=None,
        **kwargs,
    ):
        super().__init__(
            n_features=n_features,
            callbacks=callbacks,
            inverse_transform=inverse_transform,
        )

        self.avg_weight = AvgHolder([] * n_features)
        self.avg_feature = AvgHolder([] * n_features)

        self.ref_feature = []
        self.weight = []

    def get_useful_info(
        self, x: torch.FloatTensor, feature_out: List[torch.FloatTensor]
    ) -> Dict:
        return {
            "imgs": self.inverse_transform(x).detach().cpu().numpy(),
        }

    @Feature.invoke_callbacks
    @Feature.average_feature
    def __call__(self, x) -> List[torch.FloatTensor]:
        return []


@FeatureRegistry.register()
class SumFeature(Feature):
    def __init__(
        self,
        callbacks: Optional[List] = None,
        inverse_transform=None,
        **kwargs,
    ):
        super().__init__(
            n_features=0,
            callbacks=callbacks,
            inverse_transform=inverse_transform,
        )
        self.features = []

        for feature in kwargs["features"]:
            feature_kwargs = feature["params"]
            if "dis" in feature["params"]:
                feature_kwargs["dis"] = kwargs.get("dis")
            feature = FeatureRegistry.create_feature(
                feature["name"],
                inverse_transform=inverse_transform,
                **feature_kwargs,
            )
            self.features.append(feature)
            self.n_features += feature.n_features

        self.avg_feature = AvgHolder([0] * self.n_features)

    @property
    def weight(self):
        weight = []
        for feature in self.features:
            weight.extend(feature.weight)
        return weight

    @Feature.invoke_callbacks
    @Feature.average_feature
    def __call__(self, x: torch.FloatTensor):
        outs = []
        for feature in self.features:
            outs.extend(feature(x))
        return outs

    def get_useful_info(
        self, x: torch.FloatTensor, feature_out: List[torch.FloatTensor]
    ) -> Dict:
        info = {}
        k = 0
        for feature in self.features:
            info.update(
                feature.get_useful_info(
                    x, feature_out[k : k + feature.n_features]
                )
            )
            k += feature.n_features
        return info

    def weight_up(self, out: List[torch.FloatTensor], step: float):
        k = 0
        for feature in self.features:
            feature.weight_up(out[k : k + feature.n_features], step)
            k += feature.n_features
