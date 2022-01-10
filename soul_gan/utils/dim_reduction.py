import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import umap
from openTSNE import TSNE, TSNEEmbedding
from sklearn.decomposition import PCA
from torchvision import datasets
from torchvision import transforms as T

from soul_gan.datasets.utils import get_dataset
from soul_gan.utils.general_utils import DATA_DIR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10"]
    )
    parser.add_argument(
        "--method", type=str, default="pca", choices=["pca", "tsne", "umap"]
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--images", type=str, nargs="+")
    parser.add_argument("--save_dirs", type=str, nargs="+")

    args = parser.parse_args()
    return args


def main(args):
    model_path = Path(DATA_DIR, args.dataset, f"{args.method}_result.pkl")

    if args.train:
        data = get_dataset(args.dataset, mean=(0, 0, 0), std=(1, 1, 1))
        data = np.stack(
            [data[i] for i in np.random.choice(np.arange(len(data)), 10000)], 0
        )  # len(data))]
        # if args.dataset == "cifar10":
        #     data = datasets.CIFAR10(root=Path(DATA_DIR, "cifar10").as_posix())
        #     data = (
        #         data.data[np.random.choice(np.arange(len(data)), len(data))]
        #         / 255.0
        #     )
        data = data.transpose(0, 3, 1, 2)
        data = data.reshape(data.shape[0], -1)

        # else:
        #     raise NotImplementedError

        if args.method == "pca":
            model = PCA(n_components=args.dim, whiten=False)
            data_new = model.fit_transform(data)
        elif args.method == "tsne":
            model = TSNE(n_components=args.dim, verbose=True, n_jobs=8)
            model = model.fit(data)
            data_new = model[:, : args.dim]
        elif args.method == "umap":
            model = umap.UMAP(n_components=args.dim, verbose=True)
            data_new = model.fit_transform(data)
        else:
            raise NotImplementedError

        pickle.dump(model, model_path.open("wb"))
        np.save(Path(DATA_DIR, "cifar10", f"{args.method}.npy"), data_new)

    model = pickle.load(model_path.open("rb"))

    if args.images and args.save_dirs:
        for img_path, save_dir in zip(args.images, args.save_dirs):
            images = np.load(img_path)
            images = images.reshape(images.shape[0], -1)
            compressed_images = model.transform(images)[:, : args.dim]
            Path(save_dir).mkdir(exist_ok=True)
            save_path = Path(save_dir, f"{Path(img_path).stem}_{args.method}")
            np.save(save_path, compressed_images)
            pickle.dump(
                model,
                Path(
                    save_dir, f"{Path(img_path).stem}_{args.method}.pkl"
                ).open("wb"),
            )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
