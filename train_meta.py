import argparse
import datetime
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
from torch.optim import Adam  # noqa: F401
from torch.utils.data import DataLoader

from maxent_gan.datasets.utils import TrainGANDataset, get_dataset
from maxent_gan.distribution import DistributionRegistry
from maxent_gan.feature.utils import create_feature
from maxent_gan.sample import MaxEntSampler
from maxent_gan.train.loss import LossRegistry
from maxent_gan.train.trainer import Trainer
from maxent_gan.utils.callbacks import CallbackRegistry
from maxent_gan.utils.general_utils import DotConfig, random_seed, seed_worker


sys.path.append("studiogan")  # noqa: E402
from maxent_gan.models.studiogans import (  # noqa: F401, E402  isort: skip
    StudioDis,  # noqa: F401, E402  isort: skip
    StudioGen,  # noqa: F401, E402  isort: skip
)  # noqa: F401, E402  isort: skip
from maxent_gan.models.utils import GANWrapper  # noqa: E402  isort: skip


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs="+")
    parser.add_argument("--group", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--step_size", type=float)
    parser.add_argument("--weight_step", type=float)
    parser.add_argument("--feature_version", type=int)
    # parser.add_argument("--sweet_init", action="store_true")
    parser.add_argument("--sample_steps", type=int)
    # parser.add_argument(
    #     "--kernel",
    #     type=str,
    #     choices=["GaussianKernel", "LinearKernel", "PolynomialKernel"],
    # )
    parser.add_argument("--suffix", type=str)

    args = parser.parse_args()
    return args


def reset_anchors(args: argparse.Namespace, params: yaml.YAMLObject):
    if args.weight_step is not None:
        params["weight_step"] = yaml.scalarfloat.ScalarFloat(
            args.weight_step,
            prec=1,
            width=10,
            anchor=params["weight_step"].anchor.value,
        )
    if args.step_size is not None:
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
    # if args.kernel:
    #     params["kernel"] = yaml.scalarstring.LiteralScalarString(
    #         args.kernel,
    #         anchor=params["kernel"].anchor.value,
    #     )
    # if args.sweet_init:
    #     params["sweet_init"] = yaml.scalarstring.LiteralScalarString(
    #         args.sweet_init,
    #         anchor=params["sweet_init"].anchor.value,
    #     )
    if args.resume:
        if "resume" in params:
            params["resume"] = yaml.scalarstring.LiteralScalarString(
                args.resume,
                anchor=params["resume"].anchor.value,
            )
        else:
            params["resume"] = args.resume
    if args.suffix:
        params["suffix"] = yaml.scalarstring.LiteralScalarString(args.suffix)
    # if args.seed is not None:
    #     params["seed"] = yaml.scalarint.ScalarInt(
    #         args.sample_steps,
    #         anchor=params["seed"].anchor.value,
    #     )


def main(config: DotConfig, device: torch.device, group: str):
    suffix = f"_{config.suffix}" if config.suffix else ""
    dir_suffix = f"_{config.distribution.name}_train"

    if config.train_params.sub_dir:
        save_dir = Path(
            config.train_params.save_dir + dir_suffix,
            config.train_params.sub_dir + suffix,
        )
    else:
        save_dir = Path(
            config.train_params.save_dir + dir_suffix,
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + suffix,
        )
    if config.seed:
        save_dir = save_dir.with_name(f"{save_dir.name}_{config.seed}")
    save_dir.mkdir(exist_ok=True, parents=True)

    random_seed(config.seed)
    train_config = config.train_params

    dataset_stuff = get_dataset(
        config.gan_config.dataset.name,
        mean=config.gan_config.train_transform.Normalize.mean,
        std=config.gan_config.train_transform.Normalize.std,
        seed=config.seed,
        **config.gan_config.dataset.params,
    )
    dataset = TrainGANDataset(
        dataset_stuff["dataset"], train_config.n_train_iters * config.train_batch_size
    )
    g = torch.Generator()
    g.manual_seed(config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=1,
        worker_init_fn=seed_worker,
        generator=g,
    )

    random_seed(config.seed)
    print(config.resume, (save_dir / "checkpoints").exists())
    if config.resume and (save_dir / "checkpoints").exists():
        config.gan_config.generator["ckpt_path"] = sorted(
            list((save_dir / "checkpoints").glob("g_*.pth"))
        )[-1]
        config.gan_config.discriminator["ckpt_path"] = sorted(
            list((save_dir / "checkpoints").glob("d_*.pth"))
        )[-1]
        start_iter = (
            int(re.findall(r"\d+", config.gan_config.generator.ckpt_path.name)[-1]) + 1
        )
    else:
        config["resume"] = False
        start_iter = 1
    print(f"Start from iter: {start_iter}")
    gan = GANWrapper(
        config.gan_config,
        device=device,
        load_weights=config.resume,
        # eval=False
    )
    gan.gen.train()
    gan.dis.train()
    for p in list(gan.gen.parameters()) + list(gan.dis.parameters()):
        p.requires_grad_(True)

    criterion_g = LossRegistry.create(train_config.criterion_g.name)
    criterion_d = LossRegistry.create(train_config.criterion_d.name)

    optimizer_g = eval(train_config.optimizer_g.name)(
        gan.gen.parameters(), **train_config.optimizer_g.params
    )
    optimizer_d = eval(train_config.optimizer_d.name)(
        gan.dis.parameters(), **train_config.optimizer_d.params
    )

    train_callbacks = []
    for _, callback in config.callbacks.train_callbacks.items():
        params = callback.params.dict
        if "gan" in params:
            params["gan"] = gan
        if "device" in params:
            params["device"] = device
        if "np_dataset" in params:
            np_dataset = np.concatenate(
                [gan.inverse_transform(batch).numpy() for batch in dataloader], 0
            )
            params["np_dataset"] = np_dataset
        if "save_dir" in params:
            params["save_dir"] = save_dir
        if "modes" in params:
            params["modes"] = dataset_stuff["modes"]
        if "resume" in params:
            params["resume"] = config.resume
        train_callbacks.append(CallbackRegistry.create(callback.name, **params))

    ref_dist = DistributionRegistry.create(
        config.sample_params.distribution.name, gan=gan
    )
    feature_dataloader = DataLoader(dataset, batch_size=config.data_batch_size)
    feature = create_feature(
        config, gan, feature_dataloader, dataset_stuff, save_dir, device
    )
    sampler = MaxEntSampler(
        gen=gan.gen,
        ref_dist=ref_dist,
        feature=feature,
        **config.sample_params.params,
    )

    trainer = Trainer(
        gan,
        optimizer_g,
        optimizer_d,
        criterion_g,
        criterion_d,
        dataloader,
        device=device,
        callbacks=train_callbacks,
        sampler=sampler,
        start_iter=start_iter,
        **train_config.trainer_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_arguments()
    params = yaml.round_trip_load(Path(args.configs[0]).open("r"))
    reset_anchors(args, params)
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

    if args.seed is not None:
        config.seed = args.seed
    if args.sample_steps is not None:
        config.sample_params.params["n_steps"] = args.sample_steps

    config.file_name = Path(args.configs[0]).name
    config.resume = args.resume

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    group = args.group if args.group else f"{Path(args.configs[0]).stem}"
    main(config, device, group)
