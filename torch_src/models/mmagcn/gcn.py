"""
Part of code taken from https://github.com/tkipf/pygcn
Kipf, T., & Welling, M. (2016). Semi-Supervised Classification with Graph Convolutional Networks
"""

import torch
import torch.nn as nn
from typing import Union


class GCN(nn.Module):
    def __init__(self, adj: Union[torch.Tensor, torch.sparse.Tensor], in_features: int, hidden_features: int,
                 num_classes: int, dropout: float = 0.):
        super().__init__()

        self.gc1 = GraphConvolution(in_features, hidden_features, adj)
        self.gc2 = GraphConvolution(hidden_features, num_classes, adj)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)

        return x


def sparse_matmul(sparse_mat: torch.sparse.Tensor, batched_mat: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sparse.mm(sparse_mat, mat) for mat in batched_mat])


class GraphConvolution(nn.Linear):
    def __init__(self, in_features: int, out_features: int, adj: Union[torch.Tensor, torch.sparse.Tensor],
                 bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.adj = adj
        self.adj_mul = sparse_matmul if (type(adj) is torch.sparse.Tensor) else torch.matmul

    def forward(self, x):
        support = torch.matmul(x, self.weight.t())
        output = self.adj_mul(self.adj, support)
        if self.bias is not None:
            output += self.bias
        return output
