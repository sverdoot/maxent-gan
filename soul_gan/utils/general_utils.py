import random
from collections import Mapping

import numpy as np
import torch


class DotConfig(Mapping):
    """
    Simple wrapper for config
    allowing access with dot notation
    """

    def __init__(self, yaml):
        self._dict = dict(yaml)

    def __getattr__(self, key):
        if key in self.__dict__:
            return super().__getattr__(key)
        if key in self._dict:
            value = self._dict[key]
            if isinstance(value, dict):
                return DotConfig(value)
            return value
        else:
            return None

    def items(self):
        return [(k, DotConfig(v)) for k, v in self._dict.items()]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return self._dict.__iter__()

    def __getitem__(self, key):
        return self._dict[key]

    @property
    def dict(self):
        return self._dict

    def __contains__(self, key):
        return key in self._dict

    def __setitem__(self, key, value):
        self._dict[key] = value

    # def __setattr__(self, key: str, value):
    #     self.__setitem__(key, value)


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
