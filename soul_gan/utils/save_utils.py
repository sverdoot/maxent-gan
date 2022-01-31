from pathlib import Path
from typing import Union

import numpy as np


def save_latents(latents: np.ndarray, path: Union[str, Path], format: str = "npy"):
    path = Path(path)

    raise NotImplementedError


def save_weights():
    raise NotImplementedError
