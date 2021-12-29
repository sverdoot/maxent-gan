import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import ruamel.yaml as yaml

# from pytorch_fid.fid_score import calculate_frechet_distance
# from pytorch_fid.inception import InceptionV3

sys.path.append("thirdparty/studiogan/studiogan")

import wandb
from soul_gan.distribution import GANTarget
from soul_gan.feature import FeatureRegistry
from soul_gan.models.utils import load_gan
from soul_gan.models.studiogans import StudioDis, StudioGen
from soul_gan.sample import soul
from soul_gan.utils.callbacks import CallbackRegistry
from soul_gan.utils.general_utils import DotConfig  # isort:block
from soul_gan.utils.general_utils import IgnoreLabelDataset, random_seed
from soul_gan.utils.metrics.compute_fid_tf import calculate_fid_given_paths
from soul_gan.utils.metrics.inception_score import (
    MEAN_TRASFORM,
    STD_TRANSFORM,
    N_GEN_IMAGES,
    get_inception_score,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs="+")
    parser.add_argument("--thermalize", action='store_true')
    parser.add_argument("--group", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--lipschitz_step_size', action='store_true')

    args = parser.parse_args()
    return args


def main(config: DotConfig, device: torch.device, group: str):
    gen, dis = load_gan(config.gan_config, device, thermalize=config.thermalize)
    # HACK
    suffix = f'{"_therm" if config.thermalize else ""}'

    # sample
    if config.sample_params.sample:

        

        if config.sample_params.sub_dir:
            save_dir = Path(
                config.sample_params.save_dir, config.sample_params.sub_dir+suffix
            )
        else:
            save_dir = Path(
                config.sample_params.save_dir,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")+suffix,
            )
        save_dir.mkdir(exist_ok=True, parents=True)

        yaml.round_trip_dump(
            config.dict, Path(save_dir, config.file_name).open("w")
        )
        yaml.round_trip_dump(
            dict(gan_config=config.gan_config.dict),
            Path(save_dir, "gan_config.yml").open("w"),
        )

        feature_callbacks = []

        callbacks = config.callbacks.feature_callbacks

        if callbacks:
            for _, callback in callbacks.items():
                params = callback.params.dict
                # HACK
                if "dis" in params:
                    params["dis"] = dis
                if "gen" in params:
                    params["gen"] = gen
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
            "fid" in callbacks
            and config.sample_params.feature.name  # noqa: W503
            == "InceptionV3MeanFeature"  # noqa: W503
        ):
            idx = callbacks.keys().index("fid")
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
        total_sample_z = []
        total_sample_x = []
        total_labels = []
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
            # HACK
            label = torch.LongTensor(np.random.randint(0, 10 - 1, len(z))).to(
                z.device
            )
            gen.label = label
            dis.label = label

            if config.lipschitz_step_size:
                config.sample_params.params.step_size = 1. / config.gan_config.thermalize.dict[config.thermalize]['lipschitz_const']

            zs, xs = soul(
                z, gen, ref_dist, feature, **config.sample_params.params
            )
            zs = torch.stack(zs, 0)
            xs = torch.stack(xs, 0)
            print(zs.shape)
            total_sample_z.append(zs)
            total_sample_x.append(xs)
            total_labels.append(label.cpu())

        total_sample_z = torch.cat(
            total_sample_z, 1
        )  # (number_of_steps / every) x total_n x latent_dim
        total_sample_x = torch.cat(
            total_sample_x, 1
        )  # (number_of_steps / every) x total_n x 32 x 32
        total_labels = torch.cat(
            total_labels, 0
        )

        imgs_dir = Path(save_dir, "images")
        imgs_dir.mkdir(exist_ok=True)
        for slice_id in range(total_sample_x.shape[0]):
            np.save(
                Path(
                    imgs_dir,
                    f"{slice_id * config.sample_params.save_every}.npy",
                ),
                total_sample_x[slice_id].cpu().numpy(),
            )

        latents_dir = Path(save_dir, "latents")
        latents_dir.mkdir(exist_ok=True)
        for slice_id, slice in enumerate(total_sample_z):
            np.save(
                Path(
                    latents_dir,
                    f"{slice_id * config.sample_params.save_every}.npy",
                ),
                slice.cpu().numpy(),
            )

        np.save(
            Path(
                save_dir,
                f"labels.npy",
            ),
            total_labels.cpu().numpy(),
        )

    # afterall

    results_dir = config.afterall_params.results_dir
    if config.afterall_params.sub_dir == "latest":
        results_dir = filter(Path(results_dir).glob("*"))[-1]
    else:
        results_dir = Path(results_dir, config.afterall_params.sub_dir+suffix)

    assert Path(results_dir).exists()

    if config.afterall_params.init_wandb:
        wandb.init(**config.wandb_init_params, group=group)
        wandb.run.config.update({"group": f"{group}"})

    if config.afterall_params.compute_is:
        transform = torchvision.transforms.Normalize(
            MEAN_TRASFORM, STD_TRANSFORM
        )
        model = torchvision.models.inception_v3(
            pretrained=True, transform_input=False
        ).to(device)
        model.eval()

        is_values = []
        for step in range(0, config.n_steps + 1, config.every):
            file = Path(
                results_dir,
                "images",
                f"{step}.npy",
            )
            print(file)

            images = np.load(file)
            dataset = transform(torch.from_numpy(images))
            print(dataset.shape)
            dataset = IgnoreLabelDataset(
                torch.utils.data.TensorDataset(dataset)
            )

            inception_score_mean, inception_score_std, _ = get_inception_score(
                dataset, model, resize=True, device=device, batch_size=50, splits=len(images) // N_GEN_IMAGES
            )

            print(f"Iter: {step}\t IS: {inception_score_mean}")
            if wandb.run is not None:
                wandb.run.log({"step": step, "overall IS": inception_score_mean})

            is_values.append((inception_score_mean, inception_score_std))
            np.savetxt(
                Path(
                    results_dir,
                    "is_values.txt",
                ),
                is_values,
            )

    if config.afterall_params.compute_fid:
        # model = InceptionV3().to(device)
        # model.eval()
        # stats = np.load("stats/fid_stats_cifar10_train.npz")

        fid_values = []
        for step in range(0, config.n_steps + 1, config.every):
            file = Path(
                results_dir,
                "images",
                f"{step}.npy",
            )
            print(file)
            images = np.load(file)
            dataset = torch.from_numpy(images)
            print(dataset.shape)
            dataset = IgnoreLabelDataset(
                torch.utils.data.TensorDataset(dataset)
            )

            # tf version
            fid = calculate_fid_given_paths(
                ("stats/fid_stats_cifar10_train.npz", file.as_posix()),
                inception_path="thirdparty/TTUR/inception_model",
            )

            # torch version
            # mu, sigma, _ = get_activation_statistics(
            #     dataset, model, batch_size=100, device=device, verbose=True
            # )
            # fid = calculate_frechet_distance(
            #     mu, sigma, stats["mu"], stats["sigma"]
            # )
            print(f"Iter: {step}\t Fid: {fid}")
            if wandb.run is not None:
                wandb.run.log({"step": step, "overall fid": fid})

            fid_values.append(fid)
            np.savetxt(
                Path(
                    results_dir,
                    "fid_values.txt",
                ),
                fid_values,
            )

    if config.callbacks.afterall_callbacks:
        # z_final_file = Path(
        #         results_dir,
        #         config.afterall_params.sub_dir,
        #         "latents",
        #         f"{config.n_steps}.npy",
        #     )
        # label_file = Path(
        #         results_dir,
        #         config.afterall_params.sub_dir,
        #         f"labels.npy",
        #     )
        # z_final = np.load(z_final_file)
        # label = np.load(label_file)

        label_file = Path(
                results_dir,
                f"labels.npy",
            )
        label = np.load(label_file)

        afterall_callbacks = []
        callbacks = config.callbacks.afterall_callbacks
        for _, callback in callbacks.items():
            params = callback.params.dict
            # HACK
            if "dis" in params:
                params["dis"] = dis
            if "gen" in params:
                params["gen"] = gen
            afterall_callbacks.append(
                CallbackRegistry.create_callback(callback.name, **params)
            )

        results = [[] for _ in afterall_callbacks]
        for step in range(0, config.n_steps + 1, config.every):
            x_file = Path(
                results_dir,
                "images",
                f"{step}.npy",
            )
            z_file = Path(
                results_dir,
                "latents",
                f"{step}.npy",
            )
            
            print(x_file)

            images = np.load(x_file)
            zs = np.load(z_file)

            info = dict(imgs=images, zs=zs, step=step, label=label)

            for callback_id, callback in enumerate(afterall_callbacks):
                val = callback.invoke(info)
                if val is not None:
                    results[callback_id].append(val)
        results = np.array(results)
        np.savetxt(
            Path(
                results_dir,
                "callback_results.txt",
            ),
            results,
        )


if __name__ == "__main__":
    args = parse_arguments()
    import subprocess

    proc = subprocess.Popen(["cat", *args.configs], stdout=subprocess.PIPE)
    config = yaml.round_trip_load(proc.stdout.read())

    config = DotConfig(config)
    if args.seed:
        config.seed = args.seed
    config.file_name = Path(args.configs[0]).name
    config.thermalize = args.thermalize
    config.lipschitz_step_size = args.lipschitz_step_size

    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )

    group = args.group if args.group else f"{Path(args.configs[0]).stem}"
    main(config, device, group)
