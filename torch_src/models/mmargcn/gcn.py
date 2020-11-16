"""
Code based partly on https://github.com/tkipf/pygcn and https://github.com/yysijie/st-gcn.git
[1] Kipf, T., & Welling, M. (2016).
Semi-Supervised Classification with Graph Convolutional Networks
[2] Sijie Yan and Yuanjun Xiong and Dahua Lin (2018).
Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.
"""

import math
from typing import Union

import torch
import torch.nn as nn

from models.mmargcn.graph_convolution import STGCNGraphConvolution, AGCNGraphConvolution


class GCN(nn.Module):
    """
    This Graph Convolutional Neural Network is based on ST-GCN/AGCN but without the temporal component.
    ST-GCN expects input of shape: batch_size, num_channels, num_frames, num_graph_nodes.
    This module expects input of shape: batch_size, num_channels, num_graph_nodes.
    Therefore, most inner working modules are changed from 2D to 1D.
    """

    def __init__(self, adj: Union[torch.Tensor, torch.sparse.Tensor], data_shape: tuple,
                 num_classes: int, dropout: float = 0., sparse: bool = False, gc_model: str = "stgcn",
                 num_layers: int = 10, inner_feature_dim: int = 64, include_additional_top_layer: bool = False):
        super().__init__()

        assert num_layers >= 2

        if gc_model == "stgcn":
            gc = STGCNGraphConvolution
        elif gc_model == "agcn":
            gc = AGCNGraphConvolution
        else:
            raise ValueError(f"Model {gc_model} not supported.")

        feature_dim, num_nodes = data_shape

        self.layers = [
            gc(feature_dim, inner_feature_dim, adj, sparse=sparse, residual=False)
        ]

        if include_additional_top_layer:
            self.layers.append(gc(inner_feature_dim, inner_feature_dim, adj, sparse=sparse, dropout=dropout))

        k = 0
        for i in range(len(self.layers), num_layers):
            k += 1
            in_feature_dim = inner_feature_dim
            if k == 3:
                inner_feature_dim *= 2
                k = 0
            out_feature_dim = inner_feature_dim
            layer = gc(in_feature_dim, out_feature_dim, adj, sparse=sparse, dropout=dropout)
            self.layers.append(layer)

        self.bn = nn.BatchNorm1d(feature_dim * num_nodes)

        for layer_idx, layer in enumerate(self.layers):
            setattr(self, f"gc{layer_idx + 1}", layer)

        # self.gc1 = gc(feature_dim, inner_feature_dim, adj, sparse=sparse, residual=False)
        # self.gc2 = gc(inner_feature_dim, inner_feature_dim, adj, sparse=sparse, dropout=dropout)
        # self.gc3 = gc(inner_feature_dim, inner_feature_dim, adj, sparse=sparse, dropout=dropout)
        # self.gc4 = gc(inner_feature_dim, inner_feature_dim, adj, sparse=sparse, dropout=dropout)
        # self.gc5 = gc(inner_feature_dim, inner_feature_dim * 2, adj, sparse=sparse, dropout=dropout)
        # inner_feature_dim *= 2
        # self.gc6 = gc(inner_feature_dim, inner_feature_dim, adj, sparse=sparse, dropout=dropout)
        # self.gc7 = gc(inner_feature_dim, inner_feature_dim, adj, sparse=sparse, dropout=dropout)
        # self.gc8 = gc(inner_feature_dim, inner_feature_dim * 2, adj, sparse=sparse, dropout=dropout)
        # inner_feature_dim *= 2
        # self.gc9 = gc(inner_feature_dim, inner_feature_dim, adj, sparse=sparse, dropout=dropout)
        # self.gc10 = gc(inner_feature_dim, inner_feature_dim, adj, sparse=sparse, dropout=dropout)
        # self.gc11 = gc(256, 512, adj, sparse=sparse, dropout=dropout)
        # self.gc12 = gc(512, 512, adj, sparse=sparse, dropout=dropout)
        # self.gc12 = gc(512, 512, adj, sparse=sparse, dropout=dropout)
        # self.gc13 = gc(512, 1024, adj, sparse=sparse, dropout=dropout)
        # self.gc14 = gc(1024, 1024, adj, sparse=sparse, dropout=dropout)
        # self.gc15 = gc(1024, 1024, adj, sparse=sparse, dropout=dropout)
        self.fc = nn.Linear(inner_feature_dim, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))

    def forward(self, x):
        batch_size, feature_dim, num_nodes = x.size()
        x = torch.flatten(x, start_dim=1)
        x = self.bn(x)
        x = torch.reshape(x, (batch_size, feature_dim, num_nodes))

        for layer in self.layers:
            x = layer(x)

        # x = self.gc1(x)
        # x = self.gc2(x)
        # x = self.gc3(x)
        # x = self.gc4(x)
        # x = self.gc5(x)
        # x = self.gc6(x)
        # x = self.gc7(x)
        # x = self.gc8(x)
        # x = self.gc9(x)
        # x = self.gc10(x)
        # x = self.gc11(x)
        # x = self.gc12(x)
        # x = self.gc13(x)
        # x = self.gc14(x)
        # x = self.gc15(x)

        x = x.mean(-1)
        x = self.fc(x)
        return x
