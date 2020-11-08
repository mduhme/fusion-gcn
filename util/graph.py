import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp


class Graph:
    """
    Simple class to store a graph.
    """

    def __init__(self, edges, num_vertices=None, is_directed=False, center_joint=0):
        self.edges = np.unique(edges, axis=0)
        assert np.issubdtype(self.edges.dtype, np.integer)
        assert np.all(self.edges >= 0)
        assert self.edges.shape[1] == 2
        if num_vertices is None:
            self.num_vertices = np.max(self.edges) + 1
        else:
            assert num_vertices >= (np.max(self.edges) + 1)
            self.num_vertices = num_vertices
        self.__g = nx.Graph() if not is_directed else nx.DiGraph()
        self.__g.add_edges_from(self.edges)
        self.is_directed = is_directed
        self.center_joint = center_joint

    def as_directed(self):
        if self.is_directed:
            return self
        return Graph(self.edges, self.num_vertices, True, self.center_joint)

    def as_undirected(self):
        if not self.is_directed:
            return self
        return Graph(self.edges, self.num_vertices, False, self.center_joint)

    def with_reversed_edges(self):
        reversed_edges = [(j, i) for i, j in self.edges]
        return Graph(reversed_edges, self.num_vertices, self.is_directed, self.center_joint)

    def has_edge(self, edge):
        assert len(edge) == 2
        return np.any(np.sum(self.edges == edge, axis=1) == 2)

    def has_edges(self, edges):
        edges = np.array(edges)
        assert np.issubdtype(edges.dtype, np.integer)
        assert np.all(edges >= 0)
        o = np.zeros(len(edges), dtype=np.bool)
        for idx, edge in enumerate(edges):
            o[idx] = np.any(np.sum(self.edges == edge, axis=1) == 2)
        return o

    def __is_one_of(self, edges):
        m = np.zeros(len(self.edges), dtype=np.bool)
        for idx, edge in enumerate(self.edges):
            m[idx] = np.any(np.sum(edges == edge, axis=1) == 2)
        return m

    def with_new_edges(self, edges):
        edges = np.array(edges)
        assert np.issubdtype(edges.dtype, np.integer)
        assert np.all(edges >= 0)
        assert edges.shape[1] == 2
        return Graph(np.vstack((self.edges, edges)), center_joint=self.center_joint)

    def with_removed_edges(self, edges):
        edges = np.array(edges)
        assert np.issubdtype(edges.dtype, np.integer)
        assert np.all(edges >= 0)
        assert edges.shape[1] == 2
        return Graph(np.delete(self.edges, np.where(self.__is_one_of(edges)), axis=0), center_joint=self.center_joint)

    def get_adjacency_matrix(self):
        a = np.zeros((self.num_vertices, self.num_vertices), dtype=np.int)
        a[self.edges[:, 0], self.edges[:, 1]] = 1
        if not self.is_directed:
            a[self.edges[:, 1], self.edges[:, 0]] = 1
        return a

    def get_sparse_adjacency_matrix(self):
        data = np.ones(len(self.edges) * 2)
        e1 = self.edges[:, 0]
        e2 = self.edges[:, 1]
        if not self.is_directed:
            e1 = np.hstack((e1, self.edges[:, 1]))
            e2 = np.hstack((e2, self.edges[:, 0]))
        a = sp.coo_matrix((data, (e1, e2)), shape=(self.num_vertices, self.num_vertices), dtype=np.int)
        return a

    def get_degree_matrix(self, as_matrix=True):
        d = np.sum(self.get_adjacency_matrix(), axis=0)
        if as_matrix:
            return np.diag(d)
        return d

    @staticmethod
    def _reciprocal_degree(degree: np.ndarray, normalization: str) -> np.ndarray:
        if normalization == "symmetric":
            return np.reciprocal(np.sqrt(degree), where=degree > 0)
        return np.reciprocal(degree, where=degree > 0)

    @staticmethod
    def _normalize(adj, d_mat_inv, normalization: str):
        if normalization == "row":
            return d_mat_inv.dot(adj)
        elif normalization == "column":
            return adj.dot(d_mat_inv)
        elif normalization == "row_column":
            return d_mat_inv.dot(adj).dot(d_mat_inv)
        elif normalization == "symmetric":
            return d_mat_inv.dot(adj).dot(d_mat_inv)
        raise ValueError("Unsupported normalization: " + normalization)

    def get_normalized_adjacency_matrix(self, normalization="row", add_self_connections=False):
        adj = self.get_adjacency_matrix().astype(np.float)
        if add_self_connections:
            adj += np.eye(self.num_vertices)

        d = np.sum(adj, axis=0)
        d_inv = Graph._reciprocal_degree(d, normalization)
        d_mat_inv = np.diag(d_inv)
        return Graph._normalize(adj, d_mat_inv, normalization)

    def get_normalized_sparse_adjacency_matrix(self, normalization="row", add_self_connections=False):
        adj = self.get_sparse_adjacency_matrix().astype(np.float)
        if add_self_connections:
            adj += sp.eye(self.num_vertices)

        # noinspection PyUnresolvedReferences
        d = adj.sum(axis=0).A1
        d_inv = Graph._reciprocal_degree(d, normalization)
        d_mat_inv = sp.diags(d_inv)
        return Graph._normalize(adj, d_mat_inv, normalization)

    def get_laplacian_matrix(self):
        return self.get_degree_matrix() - self.get_adjacency_matrix()

    def eig(self):
        return np.linalg.eigh(self.get_laplacian_matrix())

    def draw(self):
        nx.draw_networkx(self.__g, with_labels=True)
        plt.show()

    def plot_eigenvectors(self):
        _, v = self.eig()
        fig, axs = plt.subplots(self.num_vertices, 1, sharex="all", sharey="all", figsize=(5, self.num_vertices * 2))
        for i in range(self.num_vertices):
            ax, ev = axs[i], v[:, i]
            ax.scatter(range(self.num_vertices), ev)
            ax.plot(range(self.num_vertices), ev)
        plt.show()

    def plot_eigenvalues(self):
        l, _ = self.eig()
        fig = plt.figure()
        plt.scatter(range(self.num_vertices), l)
        plt.show()

    def get_k_walk_connections(self, k, add_self_connections=False, labels=None):
        a = self.get_adjacency_matrix()
        if add_self_connections:
            a += np.eye(a.shape[0], dtype=a.dtype)
        ak = np.linalg.matrix_power(a, k)
        res = np.transpose(ak.nonzero())
        if labels is None:
            return ak, res
        return ak, [(labels[a], labels[b], ak[a, b]) for a, b in res]

    def __str__(self):
        return f"|V| = {self.num_vertices}; |E| = {len(self.edges)}"


def get_k_adjacency(adj: np.ndarray, k: int, with_self: bool = False, self_factor: int = 1) -> np.ndarray:
    identity = np.eye(len(adj), dtype=adj.dtype)
    if k == 0:
        return identity
    adj_k = np.minimum(np.linalg.matrix_power(adj + identity, k), 1) - np.minimum(
        np.linalg.matrix_power(adj + identity, k - 1), 1)
    if with_self:
        adj_k += (self_factor * identity)
    return adj_k
