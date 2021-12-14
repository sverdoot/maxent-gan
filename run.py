import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_fid.fid_score import calculate_frechet_distance
from yaml import Dumper, Loader

from soul_gan.distribution import GANTarget
from soul_gan.feature import FeatureRegistry
from soul_gan.models.utils import load_gan
from soul_gan.sample import soul
from soul_gan.utils.callbacks import CallbackRegistry
from soul_gan.utils.general_utils import DotConfig, random_seed
from soul_gan.utils.metrics.compute_fid import get_activation_statistics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("gan_config", type=str)
    parser.add_argument("--seed", type=int)

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
                params = callback.params.dict
                # HACK
                if "dis" in params:
                    params["dis"] = dis
                feature_callbacks.append(
                    CallbackRegistry.create_callback(callback.name, **params)
                )

        feature_kwargs = config.sample.feature.params.dict

        # HACK
        if "dis" in config.sample.feature.params:
            feature_kwargs["dis"] = dis

        feature = FeatureRegistry.create_feature(
            config.sample.feature.name,
            callbacks=feature_callbacks,
            inverse_transform=gen.inverse_transform,
            **feature_kwargs,
        )

        # HACK
        if (
            "FIDCallback" in config.callbacks
            and config.sample.feature.name  # noqa: W503
            == "InceptionV3MeanFeature"  # noqa: W503
        ):
            idx = config.callbacks.keys().index("FIDCallback")
            feature.callbacks[idx].model = feature.model

        ref_dist = GANTarget(gen, dis)

        if config.seed is not None:
            random_seed(config.seed)
        total_sample = []
        for i in range(0, config.sample.total_n, config.sample.batch_size):
            if i > 0:
                feature.reset()
            z = gen.prior.sample((config.sample.batch_size,))
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
                    latents_dir, f"{slice_id * config.sample.save_every}.npy"
                ),
                slice.cpu().numpy(),
            )

    if config.compute_fid:
        results_dir = config.compute_fid.results_dir
        if config.compute_fid.sub_dir == "latest":
            results_dir = filter(Path(results_dir).glob("*"))[-1]
        assert Path(results_dir).exists()

        from pytorch_fid.inception import InceptionV3

        from soul_gan.utils.general_utils import IgnoreLabelDataset

        model = InceptionV3().to(device)
        model.eval()

        stats = np.load("stats/fid_stats_cifar10_train.npz")

        fid_values = []
        for i, file in enumerate(range(config.n_steps // config.every + 1)):
            file = Path(
                results_dir,
                config.compute_fid.sub_dir,
                "latents",
                f"{i*config.every}.npy",
            )
            print(file)
            latents = np.load(file)
            dataset = (
                gen.inverse_transform(
                    gen(torch.from_numpy(latents).to(device))
                )
                .detach()
                .cpu()
            )
            print(dataset.shape)
            dataset = IgnoreLabelDataset(
                torch.utils.data.TensorDataset(dataset)
            )
            mu, sigma, _ = get_activation_statistics(
                dataset, model, batch_size=100, device=device, verbose=True
            )
            fid = calculate_frechet_distance(
                mu, sigma, stats["mu"], stats["sigma"]
            )
            print(f"Iter: {i * config.every}\t Fid: {fid}")

            fid_values.append(fid)
            np.savetxt(
                Path(
                    results_dir, config.compute_fid.sub_dir, "fid_values.txt"
                ),
                fid_values,
            )

    if config.compute_is:
        pass


if __name__ == "__main__":
    args = parse_arguments()

    config = DotConfig(yaml.load(Path(args.config).open("r"), Loader))
    if args.seed:
        config.seed = args.seed
    config.file_name = Path(args.config).name  # stem
    gan_config = DotConfig(yaml.load(Path(args.gan_config).open("r"), Loader))
    gan_config.file_name = Path(args.gan_config).name  # stem

    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    main(config, gan_config, device)
