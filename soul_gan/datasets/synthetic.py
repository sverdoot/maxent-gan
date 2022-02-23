from typing import Optional, Tuple

import numpy as np


def prepare_2d_gaussian_grid_data(
    sample_size: int,
    n_modes: int = 25,
    xlims: Tuple[float, float] = (-2, 2),
    ylims: Tuple[float, float] = (-2, 2),
    sigma: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sample from a mixture of gaussians located on a square grid
    """
    assert (n_modes ** 0.5).is_integer()
    dataset = []
    n_modes_per_ax = int(n_modes ** 0.5)
    mesh_modes = np.meshgrid(
        np.linspace(xlims[0], xlims[1], n_modes_per_ax),
        np.linspace(ylims[0], ylims[1], n_modes_per_ax),
    )
    modes = np.stack(mesh_modes, axis=-1).reshape(-1, 2)
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random

    dataset = modes[rng.choice(np.arange(len(modes)), sample_size, replace=True)]
    dataset += rng.standard_normal(dataset.shape) * sigma
    dataset = np.array(dataset, dtype=np.float32)
    return dataset, modes


def prepare_2d_ring_data(
    sample_size: int,
    n_modes: int = 8,
    rad: float = 2,
    sigma: float = 0.02,
    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    dataset = []
    for i in range(sample_size // n_modes):
        for j in range(n_modes):
            phi = 2 * np.pi * (j / float(n_modes))
            x = rad * np.cos(phi)
            y = rad * np.sin(phi)
            point = np.random.randn(2) * sigma
            point[0] += x
            point[1] += y
            dataset.append(point)
    dataset = np.array(dataset, dtype=np.float32)

    means = np.array([
        [
            rad * np.cos(2 * np.pi * float(n_modes)), 
            rad * np.sin(2 * np.pi * float(n_modes))
        ] for _ in range(n_modes)])

    return dataset, means
