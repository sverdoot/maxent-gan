import zipfile
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, Any
import numpy as np

import gdown
import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from torchvision import transforms as T

from soul_gan.utils.general_utils import DATA_DIR, IgnoreLabelDataset

from .stacked_mnist import stack_mnist
from .synthetic import prepare_2d_gaussian_grid_data, prepare_2d_ring_data


N_CIFAR_CLASSES = 10


def download_celeba():
    data_root = Path(DATA_DIR, "celeba")
    data_root.mkdir(exist_ok=True)

    # URL for the CelebA dataset
    url = "https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH"

    download_path = Path(data_root, "img_align_celeba.zip")
    gdown.download(url, download_path.as_posix(), quiet=False)

    with zipfile.ZipFile(download_path, "r") as ziphandler:
        ziphandler.extractall(data_root)


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = list(Path(root_dir).glob("*.jpg"))

        self.root_dir = root_dir
        self.transform = transform
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image
        img_path = Path(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert("RGB")
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img


def get_celeba_dataset(
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_size: int = 64,
) -> Dict[str, Dataset]:
    img_folder = Path(DATA_DIR, "celeba", "img_align_celeba")
    if not img_folder.exists():
        download_celeba()
    # Spatial size of training images, images are resized to this size.
    # Transformations to be applied to each individual image sample
    transform = T.Compose(
        [
            T.CenterCrop(178),  # Because each image is size (178, 218) spatially.
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )
    # Load the dataset from file and apply transformations
    celeba_dataset = CelebADataset(img_folder, transform)
    return {"dataset": celeba_dataset}


def get_cifar_dataset(
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_size: int = 32,
) -> Dict[str, Dataset]:
    dataset = datasets.CIFAR10(
        Path(DATA_DIR, "cifar10").as_posix(),
        download=True,
        transform=T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean, std)]),
    )
    dataset = IgnoreLabelDataset(dataset)
    return {"dataset": dataset}


def get_gaussians_grid_dataset(
    sample_size: int = 10000,
    mean: Tuple[float, float] = (0.0, 0.0),
    std: Tuple[float, float] = (1.0, 1.0),
    n_modes: int = 25,
    xlims: Tuple[float, float] = (-2, 2),
    ylims: Tuple[float, float] = (-2, 2),
    sigma: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, Union[Dataset, np.ndarray]]:
    dataset, modes = prepare_2d_gaussian_grid_data(
        sample_size, n_modes, xlims, ylims, sigma, seed
    )
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    dataset = IgnoreLabelDataset(
        TensorDataset((torch.from_numpy(dataset) - mean[None, :]) / std[None, :])
    )
    return {"dataset": dataset, "modes": modes}


def get_gaussians_ring_dataset(
    sample_size: int = 10000,
    mean: Tuple[float, float] = (0.0, 0.0),
    std: Tuple[float, float] = (1.0, 1.0),
    n_modes: int = 8,
    rad: float = 2,
    sigma: float = 0.02,
    seed: Optional[int] = None,
) -> Dict[str, Union[Dataset, np.ndarray]]:
    dataset, modes = prepare_2d_ring_data(sample_size, n_modes, rad, sigma, seed)
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    dataset = IgnoreLabelDataset(
        TensorDataset((torch.from_numpy(dataset) - mean[None, :]) / std[None, :])
    )
    return {"dataset": dataset, "modes": modes}


def get_stacked_mnist_dataset(
    sample_size: int = 60000,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Dict[str, Dataset]:
    tensor = stack_mnist(sample_size)
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    transform = T.Normalize(mean, std)
    tensor = transform(tensor)
    dataset = IgnoreLabelDataset(TensorDataset(tensor))
    return {"dataset": dataset}


def get_dataset(name: str = "cifar10", *args, **kwargs) -> Dict[str, Any]:
    if name == "cifar10":
        return get_cifar_dataset(*args, **kwargs)
    elif name == "gaussians_grid":
        return get_gaussians_grid_dataset(*args, **kwargs)
    elif name == "gaussians_ring":
        return get_gaussians_ring_dataset(*args, **kwargs)
    elif name == "celeba":
        return get_celeba_dataset(*args, **kwargs)
    elif name == "stacked_mnist":
        return get_stacked_mnist_dataset(*args, **kwargs)
    else:
        raise KeyError
