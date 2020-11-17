import torch.nn as nn

import models.mmargcn.imu_feature_models as imu_models
import models.mmargcn.rgb_feature_models as rgb_models
import models.mmargcn.skeleton_graph_fusion_models as skeleton_graph_fusion_models


class Model(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, mode: str, **kwargs):
        super().__init__()

        modes = {
            # ------------------------------------------------------
            # ----------------       RGB ONLY       ----------------
            # ------------------------------------------------------
            "rgb_patch_features": rgb_models.RgbPatchFeaturesModel,
            "rgb_patch_groups_features": rgb_models.RgbPatchGroupsFeaturesModel,
            "rgb_encoder_model": rgb_models.RgbEncoderModel,
            "rgb_r2p1d": rgb_models.RgbR2p1D,
            "rgb_r2p1d_agcn": rgb_models.RgbR2p1DAgcn,

            # ------------------------------------------------------
            # ----------------       IMU ONLY       ----------------
            # ------------------------------------------------------
            "imu_gcn": imu_models.ImuGCN,
            "imu_signal_image": imu_models.ImuSignalImageModel,

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
