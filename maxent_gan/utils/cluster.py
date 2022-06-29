import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from torch.utils.data import DataLoader
from tqdm import tqdm

from maxent_gan.datasets.utils import get_dataset
from maxent_gan.utils.general_utils import DATA_DIR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "celeba", "stacked_mnist"],
    )
    parser.add_argument(
        "--method", type=str, default="kmeans", choices=["kmeans", "spectral"]
    )
    parser.add_argument("--norm_mean", type=float, nargs="+", default=(0.5, 0.5, 0.5))
    parser.add_argument("--norm_std", type=float, nargs="+", default=(0.5, 0.5, 0.5))
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    return args


def main(args):
    dataset = get_dataset(args.dataset, mean=args.norm_mean, std=args.norm_std)
    shape = dataset[0].shape
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
            out = activation[0]
            shape = out.shape[1:]
            out = out.squeeze(3).squeeze(2).to(device)
            np_dataset.append(out.detach().cpu().numpy())
        np_dataset = np.concatenate(np_dataset, 0)
    else:
        np_dataset = np.concatenate(
            [
                dataset[i].unsqueeze(0).reshape(1, -1).numpy()
                for i in range(len(dataset))
            ],
            0,
        )

    centroids = np.zeros((args.n_clusters, np_dataset.shape[1]))
    ns = np.zeros(args.n_clusters)
    closest_pts = np.zeros_like(centroids)
    sigmas = np.zeros((args.n_clusters,))
    if args.method == "kmeans":
        model = MiniBatchKMeans(n_clusters=args.n_clusters)
        model.fit(np_dataset)
        centroids = model.cluster_centers_
        distances = model.transform(centroids)
        ids = np.argmin(distances, 1)
        centroids = centroids[ids]
        distances = model.transform(np_dataset)

        sigmas = np.zeros(args.n_clusters)
        for i, point in tqdm(enumerate(np_dataset)):
            label = np.argmin(distances[i])
            sigmas[label] = sigmas[label] + distances[i][label] ** 2
            if (
                distances[i][label]
                < model.transform(closest_pts[None, label])[0, label]
                or (closest_pts[label] == 0).all()
            ):
                closest_pts[label] = point
            ns[label] += 1

    centroids = centroids[sigmas != 0]
    ns = ns[sigmas != 0]
    closest_pts = closest_pts[sigmas != 0]
    sigmas = sigmas[sigmas != 0]

    sigmas /= ns
    sigmas = sigmas ** 0.5
    print(len(ns))
    name = f"{args.method}" + (f"_{args.model}" if args.model else "") + ".npz"
    save_path = Path(DATA_DIR, args.dataset, name)
    np.savez(
        save_path.open("wb"),
        centroids=centroids.reshape(-1, *shape),
        sigmas=sigmas,
        closest_pts=closest_pts.reshape(-1, *shape),
        priors=ns / sum(ns),
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
