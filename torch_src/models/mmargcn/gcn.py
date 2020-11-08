"""
Code based partly on https://github.com/tkipf/pygcn and https://github.com/yysijie/st-gcn.git
[1] Kipf, T., & Welling, M. (2016).
Semi-Supervised Classification with Graph Convolutional Networks
[2] Sijie Yan and Yuanjun Xiong and Dahua Lin (2018).
Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.
"""

import math
import torch
import torch.nn as nn
from typing import Union


class GCN(nn.Module):
    """
    This Graph Convolutional Neural Network is based on ST-GCN but without the temporal component.
    ST-GCN expects input of shape: batch_size, num_channels, num_frames, num_graph_nodes.
    This module expects input of shape: batch_size, num_channels, num_graph_nodes.
    Therefore, most inner working modules are changed from 2D to 1D.
    """

    def __init__(self, adj: Union[torch.Tensor, torch.sparse.Tensor], data_shape: tuple,
                 num_classes: int, dropout: float = 0., sparse: bool = False):
        super().__init__()
        feature_dim, num_nodes = data_shape
        self.bn = nn.BatchNorm1d(feature_dim * num_nodes)
        self.gc1 = GraphConvolution(feature_dim, 64, adj, sparse, residual=False)
        self.gc2 = GraphConvolution(64, 64, adj, sparse, dropout=dropout)
        self.gc3 = GraphConvolution(64, 64, adj, sparse, dropout=dropout)
        self.gc4 = GraphConvolution(64, 64, adj, sparse, dropout=dropout)
        self.gc5 = GraphConvolution(64, 128, adj, sparse, dropout=dropout)
        self.gc6 = GraphConvolution(128, 128, adj, sparse, dropout=dropout)
        self.gc7 = GraphConvolution(128, 128, adj, sparse, dropout=dropout)
        self.gc8 = GraphConvolution(128, 256, adj, sparse, dropout=dropout)
        self.gc9 = GraphConvolution(256, 256, adj, sparse, dropout=dropout)
        self.gc10 = GraphConvolution(256, 256, adj, sparse, dropout=dropout)
        self.gc11 = GraphConvolution(256, 512, adj, sparse, dropout=dropout)
        self.gc12 = GraphConvolution(512, 512, adj, sparse, dropout=dropout)
        self.gc12 = GraphConvolution(512, 512, adj, sparse, dropout=dropout)
        self.gc13 = GraphConvolution(512, 1024, adj, sparse, dropout=dropout)
        self.gc14 = GraphConvolution(1024, 1024, adj, sparse, dropout=dropout)
        self.gc15 = GraphConvolution(1024, 1024, adj, sparse, dropout=dropout)
        self.fc = nn.Linear(1024, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))

    def forward(self, x):
        batch_size, feature_dim, num_nodes = x.size()
        x = torch.flatten(x, start_dim=1)
        x = self.bn(x)
        x = torch.reshape(x, (batch_size, feature_dim, num_nodes))

        x = self.gc1(x)
        x = self.gc2(x)
        x = self.gc3(x)
        x = self.gc4(x)
        x = self.gc5(x)
        x = self.gc6(x)
        x = self.gc7(x)
        x = self.gc8(x)
        x = self.gc9(x)
        x = self.gc10(x)
        x = self.gc11(x)
        x = self.gc12(x)
        x = self.gc13(x)
        x = self.gc14(x)
        x = self.gc15(x)

        x = x.mean(-1)
        x = self.fc(x)
        return x


def sparse_matmul(sparse_mat: torch.sparse.Tensor, batched_mat: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sparse.mm(sparse_mat, mat) for mat in batched_mat])


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor, sparse: bool = False, bias: bool = True,
                 residual: bool = True, dropout: float = 0.):
        super().__init__()
        self.sparse = sparse
        self.conv = nn.Conv1d(in_features, out_features, 1, bias=bias)
        self.register_buffer("adj", adj)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if not residual:
            self.residual = lambda x: 0
        elif in_features == out_features:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv1d(in_features, out_features, 1),
                nn.BatchNorm1d(out_features),
            )

    def forward(self, x):
        support = self.conv(x)
        # support = torch.einsum("bij,ki->bkj", x, self.weight)
        # x0 = torch.matmul(x.permute(0, 2, 1), self.weight.t()).permute(0, 2, 1)

        if self.sparse:
            # very slow ...
            support = support.permute(0, 2, 1).contiguous()
            sp = []
            for mat in support:
                t = torch.sparse.mm(self.adj, mat)
                sp.append(t)
            output = torch.stack(sp).permute(0, 2, 1).contiguous()
        else:
            output = torch.matmul(support, self.adj.t())

        if self.dropout is not None:
            output = self.dropout(output)

        res = self.residual(x)
        output = self.relu(output + res)
        return output
