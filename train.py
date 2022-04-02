from pathlib import Path
import ruamel.yaml as yaml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import subprocess
from functools import partial
import datetime
import sys

sys.path.append("../studiogan")

from soul_gan.utils.train.trainer import Trainer
from soul_gan.utils.train.loss import LossRegistry
from soul_gan.utils.general_utils import DotConfig
from soul_gan.models.utils import GANWrapper
from soul_gan.sample import soul
from soul_gan.distribution import DistributionRegistry
from soul_gan.utils.callbacks import CallbackRegistry
from soul_gan.datasets.utils import get_dataset

from run import parse_arguments, create_feature


def main(config: DotConfig, device: torch.device, group: str):
    suffix = f"_{config.suffix}" if config.suffix else ""
    dir_suffix = f"_{config.distribution.name}_train"

    config.batch_size = 128

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
    save_dir.mkdir(exist_ok=True, parents=True)

    dataset_stuff = get_dataset(
        config.gan_config.dataset,
        mean=config.gan_config.train_transform.Normalize.mean,
        std=config.gan_config.train_transform.Normalize.std,
    )
    dataset = dataset_stuff["dataset"]
    dataloader = DataLoader(dataset, batch_size=config.data_batch_size)

    gan = GANWrapper(config.gan_config, device=device, load_weights=False)

    train_config = config.train_params
    criterion_g = LossRegistry.create(train_config.criterion_g.name)
    criterion_d = LossRegistry.create(train_config.criterion_d.name)

    optimizer_g = eval(train_config.optimizer_g.name)(
        gan.gen.parameters(), 
        lr=train_config.optimizer_g.params.lr, 
        betas=train_config.optimizer_g.params.betas)
    optimizer_d = eval(train_config.optimizer_d.name)(
        gan.dis.parameters(), 
        lr=train_config.optimizer_d.params.lr, 
        betas=train_config.optimizer_d.params.betas)

    train_callbacks = []
    for _, callback in config.callbacks.train_callbacks.items():
        params = callback.params.dict
        if "gan" in params:
                params["gan"] = gan
        if "np_dataset" in params:
            np_dataset = np.concatenate([gan.inverse_transform(batch).numpy() for batch in dataloader], 0)
            params["np_dataset"] = np_dataset
        if "save_dir" in params:
            params["save_dir"] = save_dir 
        if "modes" in params:
            params["modes"] = dataset_stuff["modes"]
        train_callbacks.append(CallbackRegistry.create(callback.name, **params))

    feature = create_feature(config, gan, dataloader, dataset, save_dir, device)
    ref_dist = DistributionRegistry.create(config.sample_params.distribution.name, gan=gan)
    sample_fn = partial(
        soul, gen=gan.gen, ref_dist=ref_dist, feature=feature, **config.sample_params.params
        )

    trainer = Trainer(
        gan, 
        optimizer_g, 
        optimizer_d, 
        criterion_g, 
        criterion_d, 
        dataloader, 
        device=device, 
        n_dis=train_config.n_dis, 
        callbacks=train_callbacks,
        sample_fn=sample_fn
        )

    trainer.train(n_epochs=train_config.n_epochs)


if __name__ == "__main__":
    args = parse_arguments()

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
    if args.kernel:
        params["kernel"] = yaml.scalarstring.LiteralScalarString(
            args.kernel,
            anchor=params["kernel"].anchor.value,
        )
    if args.dis_emb:
        params["dis_emb"] = yaml.scalarstring.LiteralScalarString(
            args.dis_emb,
            anchor=params["dis_emb"].anchor.value,
        )
    if args.suffix:
        params["suffix"] = yaml.scalarstring.LiteralScalarString(
            args.suffix
        )
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