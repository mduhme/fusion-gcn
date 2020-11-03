import abc
import functools
import inspect

import torch


class Fusion:
    @abc.abstractmethod
    def combine(self, *tensors: torch.Tensor) -> torch.Tensor:
        pass


class SumFusion(Fusion):
    def combine(self, *tensors: torch.Tensor) -> torch.Tensor:
        return functools.reduce(torch.add, tensors)


class ProductFusion(Fusion):
    def combine(self, *tensors: torch.Tensor) -> torch.Tensor:
        return functools.reduce(torch.mul, tensors)


class ConcatenateFusion(Fusion):
    def __init__(self, concatenate_dim: int):
        self._dim = concatenate_dim

    def combine(self, *tensors: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensors, dim=self._dim)


def get_fusion(fusion_type: str, **kwargs) -> Fusion:
    fusion_types = {
        "sum": SumFusion,
        "product": ProductFusion,
        "concatenate": ConcatenateFusion
    }

    if fusion_type not in fusion_types:
        raise ValueError("Unsupported fusion: " + fusion_type)

    args = inspect.getfullargspec(fusion_types[fusion_type].__init__).args
    kwargs = {k: v for k, v in kwargs.items() if k in args}

    return fusion_types[fusion_type](**kwargs)
