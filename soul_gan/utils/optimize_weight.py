import argparse
import datetime
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# from pytorch_fid.fid_score import calculate_frechet_distance
# from pytorch_fid.inception import InceptionV3

# sys.path.append("thirdparty/studiogan/studiogan")
sys.path.append("studiogan")

from soul_gan.datasets.utils import get_dataset
from soul_gan.feature import FeatureRegistry
from soul_gan.models.studiogans import StudioDis, StudioGen
from soul_gan.models.utils import load_gan
from soul_gan.utils.general_utils import DotConfig  # isort:block
from soul_gan.utils.general_utils import ROOT_DIR, random_seed


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1000)

    args = parser.parse_args()
    return args


def main(args, config: DotConfig, device: torch.device):
    gen, dis = load_gan(config.gan_config, device, thermalize=config.thermalize)

    feature_kwargs = config.sample_params.feature.params.dict
    # HACK
    if "dis" in config.sample_params.feature.params:
        feature_kwargs["dis"] = dis

    feature = FeatureRegistry.create_feature(
        config.sample_params.feature.name,
        inverse_transform=gen.inverse_transform,
        **feature_kwargs,
    )

    if config.seed is not None:
        random_seed(config.seed)

    # dataset = get_dataset(
    #     config.gan_config.dataset,
    #     mean=config.gan_config.train_transform.Normalize.mean,
    #     std=config.gan_config.train_transform.Normalize.std,
    # )
    # dataloader = DataLoader(
    #     dataset, batch_size=config.batch_size, shuffle=True
    # )

    weights = [nn.Parameter(w) for w in feature.weight]
    optimizer = torch.optim.SGD(
        weights, lr=1e-5, nesterov=True, momentum=0.9, weight_decay=0.1
    )

    def get_estimate(weights, batch_size=256):
        zs = gen.prior.sample((batch_size,))
        imgs = gen(zs)
        dgz = dis(imgs).squeeze()
        fxs = feature(imgs, zs)
        prod = 0
        for fx, weight in zip(fxs, weights):
            prod += torch.einsum("ab,b->a", fx, weight)
            print("p", prod)
            print("w", weight)
        loss = (torch.exp(-prod) * dgz).mean()
        return loss

    # for it in trange(args.n_steps):
    #     loss = get_estimate(weights, args.batch_size)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if it % 10 == 0:
    #         print(it, loss.item(), weights[0].norm().item())
    weights = torch.cat(weights, 0).detach().cpu().numpy()
    np.save(Path(args.save_path), weights)


if __name__ == "__main__":
    args = parse_arguments()
    import subprocess

    proc = subprocess.Popen(["cat", *args.configs], stdout=subprocess.PIPE)
    config = yaml.round_trip_load(proc.stdout.read())

    config = DotConfig(config)
    if args.seed:
        config.seed = args.seed
    # config.file_name = Path(args.configs[0]).name
    # config.thermalize = args.thermalize
    # config.lipschitz_step_size = args.lipschitz_step_size
    # config.resume = args.resume

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    main(args, config, device)
