import argparse
import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from yaml import Dumper, Loader

from soul_gan.distribution import GANTarget
from soul_gan.feature import FeatureRegistry
from soul_gan.models import ModelRegistry
from soul_gan.sample import soul
from soul_gan.utils.callbacks import CallbackRegistry
from soul_gan.utils.general_utils import DotConfig, random_seed


def load_gan(
    config: DotConfig, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    gen = ModelRegistry.create_model(
        config.generator.name, **config.generator.params
    ).to(device)
    state_dict = torch.load(
        Path(config.generator.ckpt_path, map_location=device)
    )
    gen.load_state_dict(state_dict)

    dis = ModelRegistry.create_model(
        config.discriminator.name, **config.discriminator.params
    ).to(device)
    state_dict = torch.load(
        Path(config.discriminator.ckpt_path, map_location=device)
    )
    dis.load_state_dict(state_dict)

    gen.eval()
    dis.eval()

    return gen, dis


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("gan_config", type=str)

    args = parser.parse_args()
    return args


def main(config, gan_config, device):
    gen, dis = load_gan(gan_config, device)

    if config.sample:

        if config.sample.sub_dir:
            save_dir = Path(config.sample.save_dir, config.sample.sub_dir)
        else:
            save_dir = Path(
                config.sample.save_dir,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"),
            )
        save_dir.mkdir(exist_ok=True, parents=True)

        yaml.dump(
            config.dict, Path(save_dir, config.file_name).open("w"), Dumper
        )
        yaml.dump(
            gan_config.dict,
            Path(save_dir, gan_config.file_name).open("w"),
            Dumper,
        )

        feature_callbacks = []
        if config.callbacks and config.callbacks.feature_callbacks:

            for _, callback in config.callbacks.feature_callbacks.items():
                feature_callbacks.append(
                    CallbackRegistry.create_callback(
                        callback.name, **callback.params
                    )
                )

        feature = FeatureRegistry.create_feature(
            config.sample.feature.name,
            callbacks=feature_callbacks,
            **config.sample.feature.params,
        )

        z_dim = gen.z_dim
        proposal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(z_dim).to(device), torch.eye(z_dim).to(device)
        )
        ref_dist = GANTarget(gen, dis, proposal)

        total_sample = []
        for i in range(0, config.sample.total_n, config.sample.batch_size):
            z = torch.randn(config.sample.batch_size, z_dim).to(device)
            zs = soul(z, gen, ref_dist, feature, **config.sample.params)
            zs = torch.stack(zs, 0)
            total_sample.append(zs)

        total_sample = torch.cat(
            total_sample, 1
        )  # (number_of_steps / every) x total_n x latent_dim

        latents_dir = Path(save_dir, "latents")
        latents_dir.mkdir(exist_ok=True)
        for slice_id, slice in enumerate(total_sample):
            np.save(
                Path(
                    latents_dir,
                    f"{slice_id * config.sample.save_every}.npy",
                    slice.numpy(),
                )
            )

    if config.compute_fid:
        pass

    if config.compute_is:
        pass


if __name__ == "__main__":
    args = parse_arguments()
    config = DotConfig(yaml.load(Path(args.config).open("r"), Loader))
    config.file_name = Path(args.config).name  # stem
    gan_config = DotConfig(yaml.load(Path(args.gan_config).open("r"), Loader))
    gan_config.file_name = Path(args.gan_config).name  # stem

    if config.seed is not None:
        random_seed(config.seed)

    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    main(config, gan_config, device)
