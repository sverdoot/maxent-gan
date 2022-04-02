from pathlib import Path

import numpy as np
import torch
from skimage.transform import rescale
from torchvision import datasets

from soul_gan.utils.general_utils import DATA_DIR


ORIG_IMG_SIZE = 28


def stack_mnist(
    num_training_sample: int = 60000, image_size: int = 64
) -> torch.FloatTensor:
    # Load MNIST images... 60K in train and 10K in test
    datasets.MNIST(root=DATA_DIR, download=True, transform=None)

    loaded = np.fromfile(
        file=Path(DATA_DIR, "MNIST/raw/train-images-idx3-ubyte"), dtype=np.uint8
    )
    trX = loaded[16:].reshape((60000, ORIG_IMG_SIZE, ORIG_IMG_SIZE, 1)).astype(np.float)

    # Form training and test using MNIST images
    ids = np.random.randint(0, trX.shape[0], size=(num_training_sample, 3))
    X_training = np.zeros(shape=(ids.shape[0], image_size, image_size, ids.shape[1]))

    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            xij = trX[ids[i, j], :, :, 0]
            xij = rescale(xij, (image_size / ORIG_IMG_SIZE, image_size / ORIG_IMG_SIZE))
            X_training[i, :, :, j] = xij

        # if i % 10000 == 0:
        #     print('i: {}/{}'.format(i, ids.shape[0]))
    X_training = X_training / 255.0
    X_training = torch.FloatTensor(X_training).permute(0, 3, 2, 1)
    return X_training
