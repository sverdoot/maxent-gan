import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("studiogan")

from soul_gan.datasets.utils import get_dataset
from soul_gan.feature import FeatureRegistry
from soul_gan.models.studiogans import StudioDis, StudioGen  # noqa: F401
from soul_gan.models.utils import GANWrapper
from soul_gan.utils.general_utils import DotConfig  # isort:block
from soul_gan.utils.general_utils import random_seed


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()
    return args


def evaluate(
    feature, dataset, batch_size: int, device, save_path: Optional[Path] = None
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    stats = defaultdict(lambda: 0)
    n = 0
    for batch in tqdm(dataloader):
        feature_result = feature.apply(batch.to(device))
        for i, feature_res in enumerate(feature_result):
            stats[i] += feature_res.mean(0).detach().cpu().numpy()
        n += 1
    for i in range(len(stats)):
        stats[i] /= n

    if save_path:
        np.savez(
            save_path.open("wb"),
            *stats.values(),
        )
    return stats


def main(config: DotConfig, device: torch.device):
    gan = GANWrapper(config.gan_config, device)

    feature_kwargs = config.sample_params.feature.params.dict
    # HACK
    if "dis" in config.sample_params.feature.params:
        feature_kwargs["dis"] = gan.dis

    feature = FeatureRegistry.create_feature(
        config.sample_params.feature.name,
        inverse_transform=gan.inverse_transform,
        **feature_kwargs,
    )

    if config.seed is not None:
        random_seed(config.seed)

    dataset = get_dataset(
        config.gan_config.dataset,
        mean=config.gan_config.train_transform.Normalize.mean,
        std=config.gan_config.train_transform.Normalize.std,
    )

    if not args.save_path:
        args.save_path = config.feature.params.ref_stats_path

    evaluate(
        feature, dataset, config.batch_size, device, save_path=Path(args.save_path)
    )


if __name__ == "__main__":
    args = parse_arguments()
    import subprocess

    proc = subprocess.Popen(["cat", *args.configs], stdout=subprocess.PIPE)
    config = yaml.round_trip_load(proc.stdout.read())

    config = DotConfig(config)
    if args.seed:
        config.seed = args.seed

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    main(config, device)
