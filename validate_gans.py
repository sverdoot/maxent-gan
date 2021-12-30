import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import ruamel.yaml
import torch

sys.path.append("thirdparty/studiogan/studiogan")

from soul_gan.distribution import estimate_log_norm_constant
from soul_gan.models.studiogans import StudioDis, StudioGen
from soul_gan.models.utils import estimate_lipschitz_const, load_gan
from soul_gan.utils.general_utils import CONFIGS_DIR, DotConfig  # isort:block


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_configs", type=str, nargs="+")
    parser.add_argument("--all", "-a", dest="all", action="store_true")
    parser.add_argument("--except", dest="exc", type=str, nargs="+")
    parser.add_argument("--n_pts_mc", type=int, default=10000)
    parser.add_argument("--n_pts_lipschitz", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--upd_config", action="store_true")

    parser.add_argument("--device", type=int)

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cpu" if args.device is None else args.device)

    if args.all:
        config_paths = [
            _.as_posix() for _ in Path(CONFIGS_DIR, "gan_configs").glob("*")
        ]
    else:
        config_paths = args.configs

    if args.exc:
        config_paths = sorted(
            list(
                set(config_paths)
                - set([Path(_).resolve().as_posix() for _ in args.exc])
            )
        )

    for config_path in config_paths:
        config_path = Path(config_path)
        print(f"Config: {config_path.name}")
        raw_config = ruamel.yaml.round_trip_load(config_path.open("r"))
        config = DotConfig(raw_config["gan_config"])

        raw_config["gan_config"]["thermalize"] = {True: {}, False: {}}

        for thermalize in [True, False]:
            print(f"Thermalize: {thermalize}")

            gen, dis = load_gan(config, device=device, thermalize=thermalize)

            log_norm_const = estimate_log_norm_constant(
                gen,
                dis,
                args.n_pts_mc,
                batch_size=args.batch_size,
                verbose=True,
            )
            print(f"\t log norm const: {log_norm_const:.3f}")

            lipschitz_const = estimate_lipschitz_const(
                gen,
                dis,
                args.n_pts_lipschitz,
                batch_size=args.batch_size,
                verbose=True,
            )
            print(f"\t lipschitz const: {lipschitz_const:.3f}")

            if args.upd_config:
                raw_config["gan_config"]["thermalize"][thermalize][
                    "log_norm_const"
                ] = log_norm_const
                raw_config["gan_config"]["thermalize"][thermalize][
                    "lipschitz_const"
                ] = lipschitz_const
        ruamel.yaml.round_trip_dump(raw_config, config_path.open("w"))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
