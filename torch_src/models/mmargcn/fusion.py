import abc
import functools
import inspect

import torch

from util.graph import Graph


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


class AverageFusion(Fusion):
    def combine(self, *tensors: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.stack(tensors, dim=-1), dim=-1)


class WeightedAverageFusion(Fusion):
    def __init__(self, weights: torch.Tensor):
        self.weights = weights

    def combine(self, *tensors: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.stack(tensors, dim=-1) * self.weights, dim=-1)


class ConcatenateFusion(Fusion):
    def __init__(self, concatenate_dim: int):
        self._dim = concatenate_dim

    def combine(self, *tensors: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensors, dim=self._dim)


def get_fusion(fusion_type: str, **kwargs) -> Fusion:
    fusion_types = {
        "sum": SumFusion,
        "product": ProductFusion,
        "concatenate": ConcatenateFusion,
        "average": AverageFusion,
        "weighted_average": WeightedAverageFusion
    }

    if fusion_type not in fusion_types:
        raise ValueError("Unsupported fusion: " + fusion_type)

    args = inspect.getfullargspec(fusion_types[fusion_type].__init__).args
    kwargs = {k: v for k, v in kwargs.items() if k in args}

    return fusion_types[fusion_type](**kwargs)


def get_skeleton_imu_fusion_graph(skeleton_graph: Graph, imu_enhanced_mode: str, num_imu_joints: int, **kwargs):
    # create new edges for imu data
    new_edges = []
    if imu_enhanced_mode == "append_center":
        # append imu joints to skeleton center joint
        center_joint = kwargs.get("center_joint", skeleton_graph.center_joint)
        new_edges.extend((skeleton_graph.num_vertices + i, center_joint) for i in range(num_imu_joints))

    elif imu_enhanced_mode == "append_right":
        # append imu joints to skeleton right wrist and right hip
        right_wrist_joint = kwargs["right_wrist_joint"]
        right_hip_joint = kwargs["right_hip_joint"]
        for i in range(num_imu_joints):
            new_edges.append((skeleton_graph.num_vertices + i, right_wrist_joint))
            new_edges.append((skeleton_graph.num_vertices + i, right_hip_joint))

    else:
        raise ValueError("Unsupported imu_enhanced_mode: " + imu_enhanced_mode)

    if kwargs.get("interconnect_imu_joints", False):
        for i in range(num_imu_joints):
            for j in range(i + 1, num_imu_joints):
                new_edges.append((skeleton_graph.num_vertices + i, skeleton_graph.num_vertices + j))

    return skeleton_graph.with_new_edges(new_edges)
