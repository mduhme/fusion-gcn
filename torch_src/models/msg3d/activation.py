"""
Original code from https://github.com/kenziyuliu/ms-g3d

Liu, Z., Zhang, H., Chen, Z., Wang, Z., & Ouyang, W. (2020).
Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 143–152).
"""

import torch.nn as nn


def activation_factory(name, inplace=True):
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "linear" or name is None:
        return nn.Identity()
    else:
        raise ValueError("Not supported activation:", name)