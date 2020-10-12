import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from util.partition_strategy import GraphPartitionStrategy


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# noinspection PyAbstractClass
class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


# noinspection PyAbstractClass
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adj: np.ndarray, coff_embedding: int = 4,
                 num_subsets: int = 3):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_channels = inter_channels
        self.adj_b = nn.Parameter(torch.from_numpy(adj.astype(np.float32)))
        nn.init.constant_(self.adj_b, 1e-6)
        self.adj_a = Variable(torch.from_numpy(adj.astype(np.float32)), requires_grad=False)
        self.num_subsets = num_subsets

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subsets):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subsets):
            conv_branch_init(self.conv_d[i], self.num_subsets)

    def forward(self, x):
        N, C, T, V = x.size()
        adj = self.adj_a.cuda(x.get_device())
        adj = adj + self.adj_b

        y = None
        for i in range(self.num_subsets):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_channels * T)
            A2 = self.conv_b[i](x).view(N, self.inter_channels * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + adj[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


# noinspection PyAbstractClass
class SpatialTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj, stride=1, residual=True):
        super().__init__()
        self.gcn1 = SpatialGraphConv(in_channels, out_channels, adj)
        self.tcn1 = TemporalConv(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


# noinspection PyAbstractClass
class Model(nn.Module):
    def __init__(self, data_shape: tuple, num_classes: int, graph):
        super().__init__()

        # data_shape = (num_channels, num_frames, num_joints, num_persons)
        num_channels, _, num_joints, num_persons = data_shape

        strategy = GraphPartitionStrategy()
        adj = strategy.get_adjacency_matrix_array(graph)

        self.data_bn = nn.BatchNorm1d(num_persons * num_channels * num_joints)

        self.l1 = SpatialTemporalConv(3, 64, adj, residual=False)
        self.l2 = SpatialTemporalConv(64, 64, adj)
        self.l3 = SpatialTemporalConv(64, 64, adj)
        self.l4 = SpatialTemporalConv(64, 64, adj)
        self.l5 = SpatialTemporalConv(64, 128, adj, stride=2)
        self.l6 = SpatialTemporalConv(128, 128, adj)
        self.l7 = SpatialTemporalConv(128, 128, adj)
        self.l8 = SpatialTemporalConv(128, 256, adj, stride=2)
        self.l9 = SpatialTemporalConv(256, 256, adj)
        self.l10 = SpatialTemporalConv(256, 256, adj)

        self.fc = nn.Linear(256, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        num_channels_output = x.size(1)
        x = x.view(N, M, num_channels_output, -1)
        x = x.mean(3).mean(1)

        x = self.fc(x)
        return x
