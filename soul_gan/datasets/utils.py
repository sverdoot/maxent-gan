import zipfile
from pathlib import Path
from typing import Tuple

import gdown
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T

from soul_gan.utils.general_utils import DATA_DIR, IgnoreLabelDataset

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
) -> Dataset:
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
    return celeba_dataset


def get_cifar_dataset(
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_size: int = 32,
) -> Dataset:
    dataset = datasets.CIFAR10(
        Path(DATA_DIR, "cifar10").as_posix(),
        download=True,
        transform=T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean, std)]),
    )
    dataset = IgnoreLabelDataset(dataset)
    return dataset


def get_dataset(name: str = "cifar10", *args, **kwargs) -> Dataset:
    if name == "cifar10":
        return get_cifar_dataset(*args, **kwargs)
    elif name == "celeba":
        return get_celeba_dataset(*args, **kwargs)
    else:
        raise KeyError
