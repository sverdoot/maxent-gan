import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA, KernelPCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from maxent_gan.datasets.utils import get_dataset
from maxent_gan.utils.general_utils import DATA_DIR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"]
    )
    parser.add_argument("--method", type=str, default="pca", choices=["pca"])
    parser.add_argument(
        "--kernel", type=str, default="linear", choices=["linear", "rbf", "poly"]
    )
    parser.add_argument("--n_pts", type=int)
    parser.add_argument("--norm_mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--norm_std", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--n_components", type=int, default=100)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    return args


def main(args):
    dataset = get_dataset(args.dataset, mean=args.norm_mean, std=args.norm_std)
    if args.model:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if args.model == "resnet34":
            model = torchvision.models.resnet34
        elif args.model == "resnet50":
            model = torchvision.models.resnet50
        elif args.model == "resnet101":
            model = torchvision.models.resnet101
        else:
            raise ValueError(f"Version {args.model} is not available")

        model = model(pretrained=True).to(device)
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[0] = output

            return hook

        model.avgpool.register_forward_hook(get_activation("avgpool"))
        model.eval()

        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        np_dataset = []
        for batch in tqdm(dataloader):
            model(batch.to(device))
            out = activation[0].squeeze(3).squeeze(2).to(device)
            np_dataset.append(out.detach().cpu().numpy())
        np_dataset = np.concatenate(np_dataset, 0)
    else:
        np_dataset = np.concatenate(
            [
                dataset[i].unsqueeze(0).reshape(1, -1).numpy()
                for i in range(len(dataset))
            ],
            0,
        )  # .reshape(len(dataset), -1)

    if args.method == "pca":
        if args.kernel == "linear":
            pca = PCA(n_components=args.n_components, whiten=False)
            pca.fit(np_dataset)
            components = pca.components_
            mean = pca.mean_
            cov_eigs = pca.explained_variance_ ** 0.5

            name = f"{args.method}" + (f"_{args.model}" if args.model else "") + ".npz"
            save_path = Path(DATA_DIR, args.dataset, name)
            np.savez(
                save_path.open("wb"),
                components=components,
                mean=mean,
                cov_eigs=cov_eigs,
            )

        else:
            pca = KernelPCA(
                n_components=args.n_components,
                kernel=args.kernel,
                gamma=1.0 / np_dataset.shape[1],
            )
            pca.fit(
                np_dataset[np.random.choice(np.arange(len(np_dataset)), args.n_pts)]
            )

            non_zeros = np.flatnonzero(pca.eigenvalues_)
            scaled_alphas = np.zeros_like(pca.eigenvectors_)
            scaled_alphas[:, non_zeros] = pca.eigenvectors_[:, non_zeros] / np.sqrt(
                pca.eigenvalues_[non_zeros]
            )
            x = pca.X_fit_
            gamma = pca.gamma

            name = f"{args.method}" + f"_{args.kernel}"
            name += (f"_{args.model}" if args.model else "") + ".npz"

            save_path = Path(DATA_DIR, args.dataset, name)
            np.savez(
                save_path.open("wb"),
                scaled_alphas=scaled_alphas,
                x=x,
                gamma=gamma,
            )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
