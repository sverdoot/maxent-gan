import random

import numpy as np
import torch

from collections import Mapping


class DotConfig(Mapping):
    """
    Simple wrapper for config
    allowing access with dot notation
    """

    def __init__(self, yaml):
        self._dict = dict(yaml)

    def __getattr__(self, k):
        v = self._dict[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v

    def items(self):
        return [(k, DotConfig(v)) for k, v in self._dict.items()]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return self._dict.__iter__()

    def __getitem__(self, k):
        return self._dict[k]

    @property
    def dict(self):
        return self._dict

    def __contains__(self, key):
        return key in self._dict


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)