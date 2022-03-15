import torch
import numpy as np
from pathlib import Path
from torchvision import datasets
from skimage.transform import rescale

from soul_gan.utils.general_utils import DATA_DIR

IMG_SIZE = 28


def stack_mnist(num_training_sample: int = 60000, imageSize: int = 64) -> torch.FloatTensor:
    # Load MNIST images... 60K in train and 10K in test
    datasets.MNIST(root=DATA_DIR, download=True, transform=None)

    loaded = np.fromfile(file=Path(DATA_DIR, 'MNIST/raw/train-images-idx3-ubyte'), dtype=np.uint8)
    trX = loaded[16:].reshape((60000, IMG_SIZE, IMG_SIZE, 1)).astype(np.float)

    # Form training and test using MNIST images
    ids = np.random.randint(0, trX.shape[0], size=(num_training_sample, 3))
    X_training = np.zeros(shape=(ids.shape[0], imageSize, imageSize, ids.shape[1]))

    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            xij = trX[ids[i, j], :, :, 0]
            xij = rescale(xij, (imageSize / IMG_SIZE, imageSize / IMG_SIZE))
            X_training[i, :, :, j] = xij

        # if i % 10000 == 0:
        #     print('i: {}/{}'.format(i, ids.shape[0]))
    X_training = X_training / 255.
    X_training = torch.FloatTensor(X_training).permute(0, 3, 2, 1)
    return X_training
