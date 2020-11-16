import numpy as np

import util.graph


class GraphPartitionStrategy:
    """
    Implementation of partition strategies for node neighborhoods
    Initial idea from Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
    but their implementation is strange, so use implementation similar to 2s-AGCN
    """

    def __init__(self, strategy="spatial"):
        self.strategy = strategy
        assert strategy in ("uniform", "distance", "spatial")

    def get_adjacency_matrix_array(self, graph: util.graph.Graph, normalization: str = "column"):
        """
        Applies a partition strategy to the input graph and returns one or more adjacency matrices.
        :param graph: Input graph
        :param normalization: Normalization order for adjacency matrix
        :return: A numpy array of shape (K, N, N)
        where K is the number of subsets and N the number of nodes in the graph.
        """
        if self.strategy == "distance":
            # Distance strategy:
            # Neighbors are partitioned according to each nodes' distance to the root node.
            raise NotImplementedError("Distance strategy not implemented since 'spatial' seems to yield best results.")
        elif self.strategy == "spatial":
            # Spatial strategy (K = 3):
            # Neighbors are divided into three subsets:
            # 1st subset: Only the root node itself
            # 2nd subset: All nodes that are closer to the gravity center of the skeleton than the root node
            # 3rd subset: All nodes that are farther to the gravity center of the skeleton than the root node

            # This mode assumes that the edges of the given graph are already
            # oriented towards it's center using outward connections.
            # Therefore, the first matrix with self connections is just the identity matrix.
            # The second matrix stores incoming connections, so, if A[1, 0] > 0, it means there is a connection
            # from node 0 to node 1 and also that node 1 is closer to the center.
            # The third matrix stores outgoing connections, so A[0, 1] > 1, the opposite from the second matrix.
            a = np.empty((3, graph.num_vertices, graph.num_vertices))
            a[0] = np.eye(graph.num_vertices)
            a[1] = graph.as_directed().with_reversed_edges().get_normalized_adjacency_matrix(normalization)
            a[2] = graph.as_directed().get_normalized_adjacency_matrix(normalization)
            return a

        # Uniform / Uni-labeling strategy (K = 1):
        # Neighborhood of each node is treated as a single subset.
        # Therefore, this strategy only has a single adjacency matrix (since K = 1), so simply expand dimension.
        return np.expand_dims(graph.as_undirected().get_normalized_adjacency_matrix(True), axis=0)
