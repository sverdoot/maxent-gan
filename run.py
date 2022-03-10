import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torchvision


sys.path.append("studiogan")  # noqa: E402

import wandb

from soul_gan.datasets.utils import get_dataset
from soul_gan.distribution import DistributionRegistry
from soul_gan.feature import FeatureRegistry
from soul_gan.models.studiogans import StudioDis, StudioGen  # noqa: F401
from soul_gan.models.utils import GANWrapper
from soul_gan.sample import soul
from soul_gan.utils.callbacks import CallbackRegistry
from soul_gan.utils.eval_feature import evaluate
from soul_gan.utils.general_utils import DotConfig  # isort:block
from soul_gan.utils.general_utils import IgnoreLabelDataset, random_seed
from soul_gan.utils.metrics.compute_fid_tf import calculate_fid_given_paths
from soul_gan.utils.metrics.inception_score import (
    MEAN_TRASFORM,
    N_GEN_IMAGES,
    STD_TRANSFORM,
    get_inception_score,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs="+")
    parser.add_argument("--thermalize", action="store_true")
    parser.add_argument("--group", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lipschitz_step_size", action="store_true")
    parser.add_argument("--step_size_mul", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--step_size", type=float)
    parser.add_argument("--weight_step", type=float)
    parser.add_argument("--feature_version", type=int)
    parser.add_argument("--suffix", type=str)

    args = parser.parse_args()
    return args


def main(config: DotConfig, device: torch.device, group: str):
    # HACK
    # suffix = (
    #     f'{"_therm" if config.thermalize else ""}'
    #     + f'{"_lips" if config.lipschitz_step_size else ""}'
    # )
    suffix = f"_{config.suffix}" if config.suffix else ""
    dir_suffix = f"_{config.distribution.name}"

    # sample
    if config.sample_params.sample:
        gan = GANWrapper(config.gan_config, device)

        if config.sample_params.sub_dir:
            save_dir = Path(
                config.sample_params.save_dir + dir_suffix,
                config.sample_params.sub_dir + suffix,
            )
        else:
            save_dir = Path(
                config.sample_params.save_dir + dir_suffix,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + suffix,
            )
        save_dir.mkdir(exist_ok=True, parents=True)

        yaml.round_trip_dump(config.dict, Path(save_dir, config.file_name).open("w"))
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
                    params["dis"] = gan.dis
                if "gen" in params:
                    params["gen"] = gan.gen
                if "save_dir" in params:
                    params["save_dir"] = save_dir
                feature_callbacks.append(
                    CallbackRegistry.create_callback(callback.name, **params)
                )

        feature_kwargs = config.sample_params.feature.params.dict

        # HACK
        if "dis" in config.sample_params.feature.params:
            feature_kwargs["dis"] = gan.dis

        feature = FeatureRegistry.create_feature(
            config.sample_params.feature.name,
            callbacks=feature_callbacks,
            inverse_transform=gan.inverse_transform,
            **feature_kwargs,
        )

        if config.sample_params.feature.params.ref_stats_path:
            dataset = get_dataset(
                config.gan_config.dataset,
                mean=config.gan_config.train_transform.Normalize.mean,
                std=config.gan_config.train_transform.Normalize.std,
            )
            # print(config.sample_params.feature.params.ref_stats_path)
            feature.eval = True
            stats = evaluate(
                feature,
                dataset,
                config.batch_size,
                device,
                Path(config.sample_params.feature.params.ref_stats_path),
            )
            print(stats)
            feature = FeatureRegistry.create_feature(
                config.sample_params.feature.name,
                callbacks=feature_callbacks,
                inverse_transform=gan.inverse_transform,
                **feature_kwargs,
            )
        feature.eval = False

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

        ref_dist = DistributionRegistry.create_distribution(
            config.sample_params.distribution.name, gan=gan
        )

        if config.seed is not None:
            random_seed(config.seed)
        total_sample_z = []
        total_sample_x = []
        total_labels = []
        weights = []

        for i in range(
            0, config.sample_params.total_n, config.sample_params.batch_size
        ):
            print(i)
            batch_size = min(
                config.sample_params.total_n - i,
                config.sample_params.batch_size,
            )
            if i > 0:
                feature.reset()
            if wandb.run is not None:
                run = wandb.run
                run.config.update({"group": f"{group}"})
                run.config.update({"name": f"{group}_{i}"}, allow_val_change=True)

            if config.resume:
                latents_dir = Path(save_dir, "latents")
                start_step_id = len(list(latents_dir.glob("*.npy")))  # - 1
                config.sample_params.params.dict["n_steps"] = int(
                    config.sample_params.params.n_steps
                ) - int((start_step_id - 1) * config.sample_params.save_every)
                z = torch.from_numpy(
                    np.load(sorted(list(latents_dir.glob("*.npy")))[-1])[
                        i : i + batch_size  # config.sample_params.batch_size
                    ]
                ).to(device)
                try:
                    label = torch.from_numpy(np.load(Path(save_dir, "labels.npy"))).to(
                        device
                    )
                except Exception:
                    label = torch.LongTensor(np.random.randint(0, 10 - 1, len(z))).to(
                        z.device
                    )
            else:
                z = gan.prior.sample((batch_size,))
                # HACK
                label = torch.LongTensor(np.random.randint(0, 10 - 1, len(z))).to(
                    z.device
                )
                start_step_id = 0

            gan.set_label(label)

            if config.lipschitz_step_size:
                config.sample_params.params.dict["step_size"] = (
                    args.step_size_mul
                    / config.gan_config.thermalize.dict[config.thermalize][
                        "lipschitz_const"
                    ]
                )

            zs, xs = soul(z, gan.gen, ref_dist, feature, **config.sample_params.params)

            zs = torch.stack(zs, 0)
            xs = torch.stack(xs, 0)
            print(zs.shape)
            total_sample_z.append(zs)
            total_sample_x.append(xs)
            total_labels.append(label.cpu())
            if len(feature.weight) > 0:
                weights.append(torch.cat(feature.weight, dim=0).detach())

        total_sample_z = torch.cat(total_sample_z, 1)[
            :, : config.sample_params.total_n
        ]  # (number_of_steps / every) x total_n x latent_dim
        total_sample_x = torch.cat(total_sample_x, 1)[
            :, : config.sample_params.total_n
        ]  # (number_of_steps / every) x total_n x img_size x img_size
        total_labels = torch.cat(total_labels, 0)[: config.sample_params.total_n]
        if len(feature.weight) > 0:
            weights = torch.stack(weights, 0)

        imgs_dir = Path(save_dir, "images")
        imgs_dir.mkdir(exist_ok=True)
        latents_dir = Path(save_dir, "latents")
        latents_dir.mkdir(exist_ok=True)
        for slice_id in range(start_step_id, total_sample_x.shape[0]):
            np.save(
                Path(
                    imgs_dir,
                    f"{(slice_id) * config.sample_params.save_every}.npy",
                ),
                total_sample_x[slice_id].cpu().numpy(),
            )
            np.save(
                Path(
                    latents_dir,
                    f"{(slice_id) * config.sample_params.save_every}.npy",
                ),
                total_sample_z[slice_id].cpu().numpy(),
            )

        np.save(
            Path(
                save_dir,
                "labels.npy",
            ),
            total_labels.cpu().numpy(),
        )
        if len(feature.weight) > 0:
            np.save(
                Path(
                    save_dir,
                    "weights.npy",
                ),
                weights.cpu().numpy(),
            )

    # afterall
    results_dir = config.afterall_params.results_dir + dir_suffix
    if config.afterall_params.sub_dir == "latest":
        results_dir = filter(Path(results_dir).glob("*"))[-1]
    else:
        results_dir = Path(results_dir, config.afterall_params.sub_dir + suffix)

    assert Path(results_dir).exists()

    if config.resume:
        start_step_id = np.loadtxt(Path(results_dir, "fid_values.txt"))  # - 1
    else:
        start_step_id = 0

    if config.afterall_params.init_wandb:
        wandb.init(**config.wandb_init_params, group=group)
        wandb.run.config.update({"group": f"{group}"})

    if config.afterall_params.compute_is:
        transform = torchvision.transforms.Normalize(MEAN_TRASFORM, STD_TRANSFORM)
        model = torchvision.models.inception_v3(
            pretrained=True, transform_input=False
        ).to(device)
        model.eval()

        if config.resume:
            is_values = np.loadtxt(Path(results_dir, "is_values.txt")).tolist()
        else:
            is_values = []
        for step in range(
            start_step_id * config.every,
            config.n_steps - int(config.burn_in_steps) + 1,
            config.every,
        ):
            file = Path(
                results_dir,
                "images",
                f"{step}.npy",
            )
            print(file)

            images = np.load(file)
            dataset = transform(torch.from_numpy(images))
            print(dataset.shape)
            dataset = IgnoreLabelDataset(torch.utils.data.TensorDataset(dataset))

            inception_score_mean, inception_score_std, _ = get_inception_score(
                dataset,
                model,
                resize=True,
                device=device,
                batch_size=50,  # 100,
                splits=max(1, len(images) // N_GEN_IMAGES),
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

    if config.callbacks.afterall_callbacks:
        gan = GANWrapper(config.gan_config, device)

        # x_final_file = Path(
        #     results_dir,
        #     "images",
        #     f"{config.n_steps}.npy",
        # )
        label_file = Path(
            results_dir,
            "labels.npy",
        )
        try:
            label = np.load(label_file)
        except Exception:
            label = np.random.randint(0, 10 - 1, 10000)

        # need to check correctness
        # log_norm_const = harmonic_mean_estimate(
        #     dis,
        #     np.load(x_final_file),
        #     label,
        #     device,
        #     batch_size=config.batch_size,
        # )

        afterall_callbacks = []
        callbacks = config.callbacks.afterall_callbacks
        for _, callback in callbacks.items():
            params = callback.params.dict
            # HACK
            if "dis" in params:
                params["dis"] = gan.dis
            if "gen" in params:
                params["gen"] = gan.gen
            # if "log_norm_const" in params:
            #     params["log_norm_const"] = log_norm_const
            afterall_callbacks.append(
                CallbackRegistry.create_callback(callback.name, **params)
            )

        if config.resume:
            results = np.loadtxt(Path(results_dir, "callback_results.txt")).tolist()
        else:
            results = [[] for _ in afterall_callbacks]  # []

        for step in range(
            start_step_id * config.every,
            config.n_steps - int(config.burn_in_steps) + 1,
            config.every,
        ):
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

    if config.afterall_params.compute_fid:
        # model = InceptionV3().to(device)
        # model.eval()
        # stats = np.load("stats/fid_stats_cifar10_train.npz")

        if config.resume:
            fid_values = np.loadtxt(Path(results_dir, "fid_values.txt")).tolist()
        else:
            fid_values = []
        for step in range(
            start_step_id * config.every,
            config.n_steps - int(config.burn_in_steps) + 1,
            config.every,
        ):
            file = Path(
                results_dir,
                "images",
                f"{step}.npy",
            )
            print(file)
            images = np.load(file)
            dataset = torch.from_numpy(images)
            print(dataset.shape)
            dataset = IgnoreLabelDataset(torch.utils.data.TensorDataset(dataset))

            # tf version
            stat_path = Path(
                "stats",
                f"{config.gan_config.dataset}",
                f"fid_stats_{config.gan_config.dataset}.npz",
            )
            fid = calculate_fid_given_paths(
                (stat_path.as_posix(), file.as_posix()),
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
    if config.afterall_params.remove_chains:
        for file_path in Path(results_dir, "images").glob("*.npy"):
            file_path.unlink()
        for file_path in Path(results_dir, "latents").glob("*.npy"):
            file_path.unlink()
        Path(results_dir, "labels.npy").unlink()


if __name__ == "__main__":
    args = parse_arguments()
    import subprocess

    params = yaml.round_trip_load(Path(args.configs[0]).open("r"))
    if args.weight_step:
        params["weight_step"] = yaml.scalarfloat.ScalarFloat(
            args.weight_step,
            prec=1,
            width=10,
            anchor=params["weight_step"].anchor.value,
        )
    if args.step_size:
        params["step_size"] = yaml.scalarfloat.ScalarFloat(
            args.step_size,
            prec=1,
            width=10,
            anchor=params["step_size"].anchor.value,
        )
    if args.feature_version:
        params["version"] = yaml.scalarstring.LiteralScalarString(
            args.feature_version,
            anchor=params["version"].anchor.value,
        )
    if args.suffix:
        params["suffix"] = yaml.scalarstring.LiteralScalarString(
            args.suffix
        )  # , anchor=params['suffix'].anchor.value)
    print(yaml.round_trip_dump(params))

    proc = subprocess.Popen("/bin/bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate(
        (
            " ".join(
                [
                    "echo",
                    '"' + str(yaml.round_trip_dump(params)) + '"',
                    "|",
                    "cat - ",
                    *args.configs[1:],
                ]
            )
        ).encode("utf-8")
    )
    config = yaml.round_trip_load(out.decode("utf-8"))
    config = DotConfig(config)
    if args.seed:
        config.seed = args.seed
    config.file_name = Path(args.configs[0]).name
    config.thermalize = args.thermalize
    config.lipschitz_step_size = args.lipschitz_step_size
    config.resume = args.resume

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    group = args.group if args.group else f"{Path(args.configs[0]).stem}"
    main(config, device, group)
