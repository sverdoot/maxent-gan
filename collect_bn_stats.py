import argparse
import sys
from pathlib import Path

import ruamel.yaml as yaml
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

sys.path.append("thirdparty/studiogan/studiogan")

import soul_gan.models
from soul_gan.models.studiogans import StudioDis, StudioGen
from soul_gan.models.utils import load_gan
from soul_gan.utils.general_utils import DotConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("gan_config", type=str)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--n_epoch", type=int, default=1)
    parser.add_argument("--n_d_iter", type=int, default=3)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=[
            "cifar10",
        ],
    )
    parser.add_argument("--suffix", type=str, default="upd_bn")
    parser.add_argument("--device", type=int)

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(
        args.device if args.device and torch.cuda.is_available() else "cpu"
    )
    config_path = Path(args.gan_config)
    raw_config = yaml.round_trip_load(config_path.open("r"))
    config = DotConfig(raw_config["gan_config"])

    gen, dis = load_gan(config, device, thermalize=False)
    gen.train()
    dis.train()

    def drop_to_default(net):
        from torch import nn

        for m in net.modules():
            if type(m) == nn.BatchNorm2d:
                m.reset_running_stats()

    drop_to_default(gen)
    drop_to_default(dis)

    if args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            "data/cifar10",
            train=True,
            transform=T.Compose([T.ToTensor(), dis.transform]),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

    for _ in range(args.n_epoch):
        for x_real, label_real in tqdm(dataloader):
            z = gen.prior.sample((args.batch_size,))
            label = gen.sample_label(args.batch_size, device)
            gen.label = label
            dis.label = label
            x_fake = gen(z)
            dis(x_fake)

            gen.label = label_real
            dis.label = label_real
            dis(x_real)

    gen.eval()
    dis.eval()

    gen_path = Path(config.generator.ckpt_path)
    gen_path = Path(*gen_path.parts[:-1], f"{gen_path.stem}_{args.suffix}.pth")
    torch.save(gen.state_dict(), gen_path)

    dis_path = Path(config.discriminator.ckpt_path)
    dis_path = Path(*dis_path.parts[:-1], f"{dis_path.stem}_{args.suffix}.pth")
    torch.save(dis.state_dict(), dis_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
