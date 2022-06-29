import datetime
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
from run import parse_arguments, reset_anchors
from torch.utils.data import DataLoader

from maxent_gan.datasets.utils import get_dataset
from maxent_gan.distribution import DistributionRegistry
from maxent_gan.feature.utils import create_feature
from maxent_gan.sample import MaxEntSampler
from maxent_gan.utils.callbacks import CallbackRegistry
from maxent_gan.utils.general_utils import DotConfig, random_seed, seed_worker
from maxent_gan.utils.train.loss import LossRegistry
from maxent_gan.utils.train.trainer import Trainer


sys.path.append("../studiogan")
from maxent_gan.models.utils import GANWrapper  # noqa: F401, E402  isort: skip


FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


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

    dataset_stuff = get_dataset(
        config.gan_config.dataset.name,
        mean=config.gan_config.train_transform.Normalize.mean,
        std=config.gan_config.train_transform.Normalize.std,
        seed=config.seed,
        **config.gan_config.dataset.params,
    )
    dataset = dataset_stuff["dataset"]

    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=1,
        worker_init_fn=seed_worker,
    )

    random_seed(config.seed)
    gan = GANWrapper(config.gan_config, device=device, load_weights=False)
    gan.gen.train()
    gan.dis.train()

    train_config = config.train_params
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
        train_callbacks.append(CallbackRegistry.create(callback.name, **params))

    ref_dist = DistributionRegistry.create(
        config.sample_params.distribution.name, gan=gan
    )
    if (
        config.sample_params.distribution.name == "PriorTarget"  # noqa: W503
        and config.feature.name == "DumbFeature"  # noqa: W503
    ):
        sampler = None
    else:
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
        **train_config.trainer_kwargs,
    )

    trainer.train(n_epochs=train_config.n_epochs)


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

    if args.seed:
        config.seed = args.seed
    config.file_name = Path(args.configs[0]).name
    config.resume = args.resume

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    group = args.group if args.group else f"{Path(args.configs[0]).stem}"
    main(config, device, group)
