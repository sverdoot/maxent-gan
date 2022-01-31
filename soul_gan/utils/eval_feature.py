import argparse
import datetime
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    args = parser.parse_args()
    return args


def main(config: DotConfig, device: torch.device):
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

    dataset = get_dataset(
        config.gan_config.dataset,
        mean=config.gan_config.train_transform.Normalize.mean,
        std=config.gan_config.train_transform.Normalize.std,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    stats = defaultdict(lambda: 0)
    n = 0
    for batch in tqdm(dataloader):
        feature_result = feature.apply(batch.to(device))
        for i, feature_res in enumerate(feature_result):
            stats[i] += feature_res.mean(0).detach().cpu().numpy()
        n += 1
    for i in range(len(stats)):
        stats[i] /= n
    print(stats)

    # stats_dir = Path(ROOT_DIR, "stats")
    if not args.save_path:
        args.save_path = config.feature.params.ref_stats_path
        # args.save_name = f"{config.sample_params.feature.name}_{config.gan_config.dataset}.npz"
    np.savez(
        Path(
            # stats_dir,
            args.save_path,
        ).open("wb"),
        *stats.values(),
    )


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

    main(config, device)
