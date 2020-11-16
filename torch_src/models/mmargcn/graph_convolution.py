import numpy as np
import torch
import torch.nn as nn

from models.mmargcn.agcn import conv_init, conv_branch_init, bn_init


def sparse_matmul(sparse_mat: torch.sparse.Tensor, batched_mat: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sparse.mm(sparse_mat, mat) for mat in batched_mat])


class STGCNGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor, bias: bool = True,
                 residual: bool = True, **kwargs):
        super().__init__()
        dropout = kwargs.get("dropout", 0.)
        self.sparse = kwargs.get("sparse", False)
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


# noinspection DuplicatedCode
class AGCNGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, adj, **kwargs):
        super().__init__()
        coff_embedding = kwargs.get("coff_embedding", 4)
        num_subset = kwargs.get("num_subset", 3)
        inter_channels = out_features // coff_embedding
        self.inter_c = inter_channels
        self.adj_b = nn.Parameter(torch.from_numpy(adj.astype(np.float32)))
        nn.init.constant_(self.adj_b, 1e-6)
        self.register_buffer("adj_a", torch.from_numpy(adj.astype(np.float32)))
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv1d(in_features, inter_channels, 1))
            self.conv_b.append(nn.Conv1d(in_features, inter_channels, 1))
            self.conv_d.append(nn.Conv1d(in_features, out_features, 1))

        if in_features != out_features:
            self.down = nn.Sequential(
                nn.Conv1d(in_features, out_features, 1),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm1d(out_features)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        batch_size, feature_dim, num_nodes = x.size()
        adj = self.adj_a + self.adj_b

        y = None
        for i in range(self.num_subset):
            adj_1 = self.conv_a[i](x)
            adj_1 = adj_1.permute(0, 2, 1).contiguous()
            adj_2 = self.conv_b[i](x).view(batch_size, self.inter_c, num_nodes)
            adj_1 = self.soft(torch.matmul(adj_1, adj_2) / adj_1.size(-1))  # N V V
            adj_1 = adj_1 + adj[i]
            z = self.conv_d[i](torch.matmul(x, adj_1).view(batch_size, feature_dim, num_nodes))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)
