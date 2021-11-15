"""
Code is borrowed from repository https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
"""

from typing import Iterable, Tuple
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torch.utils.data.dataset import Dataset

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from typing import Optional, Union


N_INCEPTION_CLASSES = 1000


def batch_inception(
        imgs: torch.Tensor, 
        inception_model: nn.Module,  
        resize: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    device = imgs.device
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, -1).data

    preds = get_pred(imgs)
    return preds


def get_inception_score(
        imgs: Iterable, 
        inception_model: Optional[nn.Module] = None, 
        gen: Optional[nn.Module] = None, 
        generate_from_latents: bool = False, 
        cuda: bool = True, 
        batch_size: int=32, 
        resize: bool = False, 
        splits: int = 1, 
        device: Union[torch.device, int] = 0
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    if not isinstance(imgs, torch.utils.data.Dataset):
        imgs = torch.utils.data.TensorDataset(imgs)

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    if inception_model is None:
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model.eval()
    
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x).logits
        return F.softmax(x).data.cpu()#.numpy()

    # Get predictions
    preds = torch.zeros((N, N_INCEPTION_CLASSES))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        #batchv = Variable(batch)
        if generate_from_latents:
            batchv = gen(batchv)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = torch.mean(part, 0)
        scores = []
        split_scores.append(
            torch.exp(torch.mean(
                torch.kl_div(part, torch.log(py[None, :])).sum(1)
                ))
        )

    return torch.mean(split_scores), torch.std(split_scores), preds


if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (get_inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))