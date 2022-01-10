import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from soul_gan.datasets.utils import get_dataset
from soul_gan.utils.general_utils import DATA_DIR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"]
    )
    parser.add_argument(
        "--method", type=str, default="kmeans", choices=["kmeans"]
    )
    parser.add_argument(
        "--norm_mean", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument(
        "--norm_std", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--n_clusters", type=str, default=100)
    # parser.add_argument('--savepath')

    args = parser.parse_args()
    return args


def main(args):
    dataset = get_dataset(args.dataset, mean=args.norm_mean, std=args.norm_std)
    np_dataset = np.stack(
        [dataset[i].numpy() for i in range(len(dataset))], 0
    ).reshape(len(dataset), -1)
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
        for i, point in enumerate(np_dataset):
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
    save_path = Path(DATA_DIR, args.dataset, f"{args.method}.npz")
    np.savez(
        save_path.open("wb"),
        centroids=centroids,
        sigmas=sigmas,
        closest_pts=closest_pts,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
