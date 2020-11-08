import torch
import torch.nn as nn

from models.mmargcn.gcn import GCN
from util.graph import Graph
import util.sparse as sparse_util


def build_imu_graph(data_shape: tuple, num_signals: int = 0, inter_signal_back_connections=False) -> Graph:
    sequence_length, num_signals_0 = data_shape

    assert num_signals == 0 or (num_signals_0 % num_signals) == 0
    if num_signals == 0:
        num_signals = num_signals_0

    num_vertices = sequence_length * num_signals
    graph_edges = []
    for i in range(0, num_vertices, num_signals):

        # spatial connections (connections between all values at a single time step)
        # IMU data is in form (sequence_length = [N + 1], num_signals = [M + 1]) with samples TnSm and 0<=n<=N; 0<=m<=M
        # memory layout for vertices will therefore be: T0S0, T0S1, T0S2, ... T0SM, T1S0, T1S1, ... T1Sm, ..., TNSM
        for j in range(num_signals):
            for k in range(j + 1, num_signals):
                graph_edges.append((i + j, i + k))

        # temporal back connections (connection from current value to value of same type at previous time step)
        if i > 0:
            for j in range(num_signals):
                graph_edges.append((i - num_signals + j, i + j))

            # temporal back connections (connection from current value to all other values at previous time step)
            if inter_signal_back_connections:
                for j in range(num_signals):
                    for k in range(num_signals):
                        if j == k:
                            continue
                        graph_edges.append((i - num_signals + j, i + k))

    return Graph(graph_edges, num_vertices)


def build_imu_graph_adjacency(data_shape: tuple, num_signals: int = 0, sparse=False, normalization="row",
                              build_graph_fn=build_imu_graph) -> torch.Tensor:
    graph = build_graph_fn(data_shape, num_signals, inter_signal_back_connections=False)
    if sparse:
        adj = graph.get_normalized_sparse_adjacency_matrix(normalization, True)
        return sparse_util.scipy_to_torch(adj)

    adj = graph.get_normalized_adjacency_matrix(normalization, True)
    return torch.from_numpy(adj).to(torch.float32)


class ImuGCN(nn.Module):
    def __init__(self, data_shape, num_classes: int, **kwargs):
        super().__init__()
        data_shape = data_shape["inertial"]
        dropout = kwargs.get("dropout", 0.)
        sparse = kwargs.get("sparse", False)
        num_signals = kwargs.get("num_signals", 0)
        adj = build_imu_graph_adjacency(data_shape, num_signals, sparse)
        self.gcn = GCN(adj, (1, data_shape[0] * data_shape[1]), num_classes, dropout, sparse)

    def forward(self, x) -> None:
        x = torch.flatten(x, start_dim=1).unsqueeze(1).contiguous()
        x = self.gcn(x)
        return x
