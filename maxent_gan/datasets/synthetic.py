from typing import Optional, Sequence, Tuple

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
    if not (n_modes ** 0.5).is_integer():
        raise Exception
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


def prepare_3d_gaussian_grid_data(
    sample_size: int,
    n_modes: int = 125,
    xlims: Tuple[float, float] = (-2, 2),
    ylims: Tuple[float, float] = (-2, 2),
    zlims: Tuple[float, float] = (-2, 2),
    sigma: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sample from a mixture of gaussians located on a square grid
    """
    # if not (n_modes ** 0.5).is_integer():
    #     raise Exception
    dataset = []
    n_modes_per_ax = round(n_modes ** (1 / 3))
    mesh_modes = np.meshgrid(
        np.linspace(xlims[0], xlims[1], n_modes_per_ax),
        np.linspace(ylims[0], ylims[1], n_modes_per_ax),
        np.linspace(zlims[0], zlims[1], n_modes_per_ax),
    )
    modes = np.stack(mesh_modes, axis=-1).reshape(-1, 3)#.tolist()
    # modes.remove([0, 0, 0])
    # modes = np.array(modes)
    # print(modes)
    # weights = np.exp(-np.abs(modes[:, 2]) / (xlims[1] - xlims[0]) / 0.25)
    # weights = weights / sum(weights)
    # print(weights)
    # weights = [1.33, 1, 1, 1.33, 1, 1, 0.66, 1]
    # weights = [1, 1, 1, 1, 1, 1, 1, 1]
    # weights = np.array(weights) / sum(weights)

    # weights = np.random.rand(n_modes)
    weights = np.array([2, 1, 1, 0.25, 1, 1, 2, 1])
    weights = weights / sum(weights)

    # def create_sphere(cx,cy,cz, r, resolution=360):
    #     '''
    #     create sphere with center (cx, cy, cz) and radius r
    #     '''
    #     phi = np.linspace(0, 2*np.pi, 2*resolution)
    #     theta = np.linspace(0, np.pi, resolution)

    #     theta, phi = np.meshgrid(theta, phi)

    #     r_xy = r*np.sin(theta)
    #     x = cx + np.cos(phi) * r_xy
    #     y = cy + np.sin(phi) * r_xy
    #     z = cz + r * np.cos(theta)

    #     return np.stack([x,y,z])
    
    # dataset = create_sphere(0, 0, 0, 2)
    # dataset = dataset.transpose(1, 2, 0).reshape(-1, 3)

    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random

    dataset = modes[rng.choice(np.arange(len(modes)), sample_size, replace=True, p=weights)]
    dataset += rng.standard_normal(dataset.shape) * sigma
    dataset = np.array(dataset, dtype=np.float32)
    return dataset, modes


def prepare_2d_ring_data(
    sample_size: int,
    n_modes: int = 8,
    rad: float = 2,
    sigma: float = 0.02,
    weights: Optional[Sequence] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.ones((n_modes,)) if weights is None else weights
    weights = np.array(weights) / np.sum(weights)

    means = np.array(
        [
            [
                rad * np.cos(2 * np.pi * i / float(n_modes)),
                rad * np.sin(2 * np.pi * i / float(n_modes)),
            ]
            for i in range(n_modes)
        ]
    ).astype(np.float32)

    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random

    ids = rng.choice(np.arange(n_modes), sample_size, replace=True, p=weights)
    dataset = means[ids]
    dataset += rng.standard_normal(dataset.shape) * sigma

    return dataset, means
