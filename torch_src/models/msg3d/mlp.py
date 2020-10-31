"""
Original code from https://github.com/kenziyuliu/ms-g3d

Liu, Z., Zhang, H., Chen, Z., Wang, Z., & Ouyang, W. (2020).
Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 143–152).
"""

import torch.nn as nn

from models.msg3d.activation import activation_factory


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu", dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x

