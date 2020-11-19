import torch.nn as nn

import models.mmargcn.imu_feature_models as imu_models
import models.mmargcn.rgb_feature_models as rgb_models
import models.mmargcn.early_fusion_models as early_fusion_models
import models.mmargcn.late_fusion_models as late_fusion_models


class Model(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, mode: str, **kwargs):
        super().__init__()

        modes = {
            # ------------------------------------------------------
            # ----------------       RGB ONLY       ----------------
            # ------------------------------------------------------
            "rgb_patch_features": rgb_models.RgbPatchFeaturesModel,
            "rgb_patch_groups_features": rgb_models.RgbPatchGroupsFeaturesModel,
            "rgb_encoder_model": rgb_models.RgbCnnEncoderModel,
            "rgb_r2p1d": rgb_models.RgbR2p1DModel,

            # ------------------------------------------------------
            # ----------------       IMU ONLY       ----------------
            # ------------------------------------------------------
            "imu_gcn": imu_models.ImuGCN,
            "imu_signal_image": imu_models.ImuSignalImageModel,

            # ------------------------------------------------------
            # ----------------    SKELETON + RGB    ----------------
            # ------------------------------------------------------
            "skeleton_rgb_encoding_early_fusion": early_fusion_models.SkeletonRgbEarlyFusion,
            "skeleton_rgb_encoding_r2p1d_early_fusion": early_fusion_models.SkeletonRgbR2P1DEarlyFusion,
            "skeleton_rgb_r2p1d_late_fusion": late_fusion_models.SkeletonRgbR2P1D,

            # ------------------------------------------------------
            # ----------------    SKELETON + IMU    ----------------
            # ------------------------------------------------------
            "skeleton_imu_enhanced": early_fusion_models.SkeletonImuEnhancedModel,
            "skeleton_imu_gcn_late_fusion": late_fusion_models.SkeletonImuGCN,

            # ------------------------------------------------------
            # ---------------- SKELETON + RGB + IMU ----------------
            # ------------------------------------------------------
            "skeleton_imu_rgb_graph_early_fusion": early_fusion_models.SkeletonImuRgbEarlyFusion
        }

        if mode not in modes:
            raise ValueError("Unsupported mode: " + mode)

        self._model = modes[mode](data_shape, num_classes, graph=graph, **kwargs)

    def forward(self, x):
        return self._model(x)
