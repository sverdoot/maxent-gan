from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from tqdm import tqdm

from soul_gan.utils.callbacks import Callback, CallbackRegistry


@torch.no_grad()
def get_activation_statistics(
    dataset,
    model,
    dims=2048,
    batch_size=100,
    num_workers=4,
    device=0,
    verbose=False,
):
    if len(dataset) > batch_size:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        n_img = len(dataset)
    else:
        dataloader = dataset
        n_img = sum([len(batch) for batch in dataset])

    pred_arr = np.empty((n_img, dims))

    start_idx = 0

    if verbose:
        loader = tqdm(dataloader)
    else:
        loader = dataloader

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy().astype(np.float32)
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred
        start_idx += pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)

    return mu, sigma, pred_arr


@CallbackRegistry.register()
class FIDCallback(Callback):
    def __init__(
        self,
        data_stat_path: Union[Path, str],
        invoke_every: int = 1,
        update_input=True,
        device: Union[str, int, torch.device] = "cuda",
        dims=2048,
        dp=False,
    ):
        self.invoke_every = invoke_every
        self.update_input = update_input
        self.data_stat_path = data_stat_path
        self.data_stat = np.load(Path(self.data_stat_path))
        self.device = device

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        if dp:
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

        self.dims = dims

    @torch.no_grad()
    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        score = None
        if self.cnt % self.invoke_every == 0:
            imgs = torch.from_numpy(info["imgs"]).to(self.device)
            fake_dataset = [imgs]
            fake_mu, fake_sigma = get_activation_statistics(
                fake_dataset,
                self.model,
                self.dims,
                imgs.shape[0],
                device=self.device,
            )[:2]

            score = calculate_frechet_distance(
                self.data_stat["mu"],
                self.data_stat["sigma"],
                fake_mu,
                fake_sigma,
            )

            if self.update_input:
                info["fid"] = score
        self.cnt += 1
        return score


# def compute_fid_for_latents(latents, gen):
#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

#     model = InceptionV3([block_idx]).to(device)
#     model.eval()
