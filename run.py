import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from yaml import Dumper, Loader

import wandb
from soul_gan.distribution import GANTarget
from soul_gan.feature import FeatureRegistry
from soul_gan.models.utils import load_gan
from soul_gan.sample import soul
from soul_gan.utils.callbacks import CallbackRegistry
from soul_gan.utils.general_utils import DotConfig  # isort:block
from soul_gan.utils.general_utils import IgnoreLabelDataset, random_seed
from soul_gan.utils.metrics.compute_fid import get_activation_statistics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("gan_config", type=str)
    parser.add_argument("--group", type=str)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    return args


def main(
    config: DotConfig, gan_config: DotConfig, device: torch.device, group: str
):
    if config.sample_params.sample:
        gen, dis = load_gan(gan_config, device)

        if config.sample_params.sub_dir:
            save_dir = Path(
                config.sample_params.save_dir, config.sample_params.sub_dir
            )
        else:
            save_dir = Path(
                config.sample_params.save_dir,
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

        feature_kwargs = config.sample_params.feature.params.dict

        # HACK
        if "dis" in config.sample_params.feature.params:
            feature_kwargs["dis"] = dis

        feature = FeatureRegistry.create_feature(
            config.sample_params.feature.name,
            callbacks=feature_callbacks,
            inverse_transform=gen.inverse_transform,
            **feature_kwargs,
        )

        # HACK
        if (
            "fid" in config.callbacks
            and config.sample_params.feature.name  # noqa: W503
            == "InceptionV3MeanFeature"  # noqa: W503
        ):
            idx = config.callbacks.keys().index("fid")
            feature.callbacks[idx].model = feature.model

        # # HACK
        # if (
        #     "WandbCallback" in config.callbacks
        #     and config.sample.feature.name  # noqa: W503
        #     == "InceptionV3MeanFeature"  # noqa: W503
        # ):
        #     idx = config.callbacks.keys().index("FIDCallback")
        #     feature.callbacks[idx].model = feature.model

        ref_dist = GANTarget(gen, dis)

        if config.seed is not None:
            random_seed(config.seed)
        total_sample = []
        for i in range(
            0, config.sample_params.total_n, config.sample_params.batch_size
        ):
            print(i)
            if i > 0:
                feature.reset()
            if wandb.run is not None:
                run = wandb.run
                run.config.update({"group": f"{group}"})
                run.config.update(
                    {"name": f"{group}_{i}"}, allow_val_change=True
                )

            # z = gen.prior.sample((config.sample_params.batch_size,))
            z = torch.randn((config.sample_params.batch_size, gen.z_dim)).to(
                device
            )
            zs = soul(z, gen, ref_dist, feature, **config.sample_params.params)
            zs = torch.stack(zs, 0)
            print(zs.shape)
            total_sample.append(zs)

        total_sample = torch.cat(
            total_sample, 1
        )  # (number_of_steps / every) x total_n x latent_dim

        # latents_dir = Path(save_dir, "latents")
        # latents_dir.mkdir(exist_ok=True)
        # for slice_id, slice in enumerate(total_sample):
        #     np.save(
        #         Path(
        #             latents_dir, f"{slice_id * config.sample.save_every}.npy"
        #         ),
        #         slice.cpu().numpy(),
        #     )
        imgs_dir = Path(save_dir, "images")
        imgs_dir.mkdir(exist_ok=True)
        for slice_id in range(total_sample.shape[0]):
            np.save(
                Path(
                    imgs_dir,
                    f"{slice_id * config.sample_params.save_every}.npy",
                ),
                total_sample[slice_id].cpu().numpy(),
            )

    if config.compute_fid:
        results_dir = config.compute_fid.results_dir
        if config.compute_fid.sub_dir == "latest":
            results_dir = filter(Path(results_dir).glob("*"))[-1]
        assert Path(results_dir).exists()

        model = InceptionV3().to(device)
        model.eval()

        stats = np.load("stats/fid_stats_cifar10_train.npz")

        if config.compute_fid.init_wandb:
            wandb.init(**config.wandb_init_params, group=group)
            wandb.run.config.update({"group": f"{group}"})

        fid_values = []
        for step in range(0, config.n_steps, config.every):
            file = Path(
                results_dir,
                config.compute_fid.sub_dir,
                "images",
                f"{step}.npy",
            )
            print(file)
            # latents = np.load(file)
            # dataset = (
            #     gen.inverse_transform(
            #         gen(torch.from_numpy(latents).to(device))
            #     )
            #     .detach()
            #     .cpu()
            # )
            images = np.load(file)
            dataset = torch.from_numpy(images)
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
            print(f"Iter: {step}\t Fid: {fid}")
            if wandb.run is not None:
                wandb.run.log({"step": step, "overall fid": fid})

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

    group = (
        args.group
        if args.group
        else f"{Path(args.gan_config).stem}_{Path(args.config).stem}"
    )
    main(config, gan_config, device, group)
