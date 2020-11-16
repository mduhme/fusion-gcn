import torch
import torch.nn as nn

from models.mmargcn.gcn import GCN
from util.graph import Graph
import util.sparse as sparse_util

from util.partition_strategy import GraphPartitionStrategy


def build_imu_graph(data_shape: tuple, num_signals: int = 0, temporal_back_connections: int = 1,
                    inter_signal_back_connections=False) -> Graph:
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
                graph_edges.append((i + k, i + j))

        # temporal back connections
        for j in range(min(i // num_signals, temporal_back_connections)):
            for k in range(num_signals):
                for m in range(num_signals):
                    if k == m or inter_signal_back_connections:
                        graph_edges.append((i - num_signals * (j + 1) + k, i + m))

    return Graph(graph_edges, num_vertices)


def build_imu_graph_adjacency(data_shape: tuple,
                              num_signals: int = 0,
                              gc_model: str = "stgcn",
                              sparse=False,
                              normalization="row",
                              temporal_back_connections: int = 1,
                              inter_signal_back_connections: bool = False,
                              build_graph_fn=build_imu_graph) -> torch.Tensor:
    graph = build_graph_fn(data_shape, num_signals, temporal_back_connections, inter_signal_back_connections)

    if gc_model == "agcn":
        strategy = GraphPartitionStrategy()
        return strategy.get_adjacency_matrix_array(graph)

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
        num_layers = kwargs.get("num_layers", 10)
        inner_feature_dim = kwargs.get("inner_feature_dim", 64)
        include_additional_top_layer = kwargs.get("include_additional_top_layer", False)
        num_temporal_back_connections = kwargs.get("num_temporal_back_connections", 1)
        inter_signal_back_connections = kwargs.get("inter_signal_back_connections", False)
        adjacency_normalization = kwargs.get("adjacency_normalization", "column")
        gc_model = kwargs.get("gc_model", "agcn")
        self.graph_node_format = kwargs.get("graph_node_format", "node_per_value")

        if self.graph_node_format == "node_per_value":
            num_signals = data_shape[1]
            self.num_features = 1
        elif self.graph_node_format == "node_per_sensor":
            num_signals = kwargs["num_signals"]
            self.num_features = data_shape[1] // num_signals
        else:
            raise ValueError(f"Unknown graph_node_format {self.graph_node_format}")

        num_nodes = data_shape[0] * num_signals
        adj = build_imu_graph_adjacency(data_shape, num_signals, gc_model, sparse, adjacency_normalization,
                                        num_temporal_back_connections, inter_signal_back_connections)
        self.gcn = GCN(adj, (self.num_features, num_nodes), num_classes, dropout, sparse, gc_model, num_layers,
                       inner_feature_dim, include_additional_top_layer)

    def forward(self, x) -> None:
        if self.graph_node_format == "node_per_value":
            x = x.flatten(start_dim=1).unsqueeze(1).contiguous()
        elif self.graph_node_format == "node_per_sensor":
            x = x.view(x.shape[0], -1, self.num_features).permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Unknown graph_node_format {self.graph_node_format}")

        x = self.gcn(x)
        return x


class ImuSignalImageModel(nn.Module):
    def __init__(self, data_shape, num_classes: int, **kwargs):
        super().__init__()
        data_shape = data_shape["inertial"]
