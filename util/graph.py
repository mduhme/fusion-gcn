import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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
        self.__a = None  # adjacency matrix
        self.__d = None  # degree matrix
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
        if self.__a is None:
            a = np.zeros((self.num_vertices, self.num_vertices), dtype=np.int)
            a[self.edges[:, 0], self.edges[:, 1]] = 1
            if not self.is_directed:
                a[self.edges[:, 1], self.edges[:, 0]] = 1
            self.__a = a
        return self.__a

    def get_degree_matrix(self, as_matrix=True):
        if self.__d is None:
            self.__d = np.sum(self.get_adjacency_matrix(), axis=0)
        if as_matrix:
            return np.diag(self.__d)
        return self.__d

    def get_normalized_adjacency_matrix(self, add_self_connections=False):
        adj = self.get_adjacency_matrix().astype(np.float)
        if add_self_connections:
            adj = adj + np.eye(self.num_vertices)
        d = np.sum(adj, axis=0)
        d_inv = np.zeros_like(d)
        if self.is_directed:
            np.reciprocal(d, out=d_inv, where=d > 0)
        else:
            np.reciprocal(np.sqrt(d), out=d_inv, where=d > 0)
        d_mat_inv = np.diag(d_inv)
        if self.is_directed:
            return adj.dot(d_mat_inv)
        return adj.dot(d_mat_inv).transpose().dot(d_mat_inv)

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
        return f"A =\n{self.get_adjacency_matrix()}\nD = {self.get_degree_matrix(False)}\nL =\n" \
               f"{self.get_laplacian_matrix()}\n"


def normalize_adjacency_matrix_undirected(adj: np.ndarray) -> np.ndarray:
    adj = adj.astype(np.float) + np.eye(adj.shape[0])
    d = np.sum(adj, axis=0)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def normalize_adjacency_matrix_directed(adj: np.ndarray) -> np.ndarray:
    adj = adj.astype(np.float) + np.eye(adj.shape[0])
    d = np.sum(adj, axis=0)
    d_inv_sqrt = np.power(d, -1)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return adj.dot(np.diag(d_inv_sqrt))


def get_k_adjacency(adj: np.ndarray, k: int, with_self: bool = False, self_factor: int = 1) -> np.ndarray:
    identity = np.eye(len(adj), dtype=adj.dtype)
    if k == 0:
        return identity
    adj_k = np.minimum(np.linalg.matrix_power(adj + identity, k), 1) - \
            np.minimum(np.linalg.matrix_power(adj + identity, k - 1), 1)
    if with_self:
        adj_k += (self_factor * identity)
    return adj_k
