import torch.nn as nn

import models.mmargcn.imu_feature_models as imu_models
import models.mmargcn.rgb_feature_models as rgb_models
import models.mmargcn.skeleton_graph_fusion_models as skeleton_graph_fusion_models


class Model(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, mode: str, **kwargs):
        super().__init__()

        # a1a = graph.get_sparse_adjacency_matrix()
        # a2a = graph.get_adjacency_matrix()
        # import numpy as np
        # import scipy.sparse as sp
        # def normalize(mx):
        #     """Row-normalize sparse matrix"""
        #     mx = mx.astype(np.float)
        #     rowsum = np.array(mx.sum(1))
        #     r_inv = np.power(rowsum, -1).flatten()
        #     r_inv[np.isinf(r_inv)] = 0.
        #     r_mat_inv = sp.diags(r_inv)
        #     mx = r_mat_inv.dot(mx)
        #     return mx
        # from util.graph import Graph
        # g = Graph([
        #     (0, 1),
        #     (0, 2),
        #     (1, 3),
        #     (3, 4),
        #     (1, 4)
        # ])
        # a0 = g.get_adjacency_matrix()
        # a1 = g.get_normalized_adjacency_matrix()
        # a2 = g.get_normalized_adjacency_matrix(normalization="column")
        # a3 = g.get_normalized_adjacency_matrix(normalization="symmetric")
        # a4 = g.get_normalized_adjacency_matrix(normalization="row_column")
        # # https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix
        # eig0 = np.linalg.eig(a0)
        # eig = np.linalg.eig(a3)

        modes = {
            # ------------------------------------------------------
            # ----------------       RGB ONLY       ----------------
            # ------------------------------------------------------
            "rgb_patch_features": rgb_models.RgbPatchFeaturesModel,
            "rgb_patch_groups_features": rgb_models.RgbPatchGroupsFeaturesModel,
            "rgb_encoder_model": rgb_models.RgbEncoderModel,

            # ------------------------------------------------------
            # ----------------       IMU ONLY       ----------------
            # ------------------------------------------------------
            "imu_gcn": imu_models.ImuGCN,

            # ------------------------------------------------------
            # ----------------    SKELETON + RGB    ----------------
            # ------------------------------------------------------
            "skeleton_rgb_encoding_early_fusion": skeleton_graph_fusion_models.SkeletonRgbEarlyFusion,

            # ------------------------------------------------------
            # ----------------    SKELETON + IMU    ----------------
            # ------------------------------------------------------
            "skeleton_imu_enhanced": skeleton_graph_fusion_models.SkeletonImuEnhancedModel,

            # ------------------------------------------------------
            # ---------------- SKELETON + RGB + IMU ----------------
            # ------------------------------------------------------
            "skeleton_imu_rgb_graph_early_fusion": skeleton_graph_fusion_models.SkeletonImuRgbEarlyFusion
        }

        if mode not in modes:
            raise ValueError("Unsupported mode: " + mode)

        self._model = modes[mode](data_shape, num_classes, graph=graph, **kwargs)

    def forward(self, x):
        return self._model(x)
