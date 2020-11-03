import torch.nn as nn

import models.mmagcn.rgb_features_model as rgb_models
from models.mmagcn.skeleton_imu_enhanced_model import SkeletonImuEnhancedModel


# noinspection PyAbstractClass
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

            # ------------------------------------------------------
            # ----------------    SKELETON + RGB    ----------------
            # ------------------------------------------------------
            "skeleton_rgb_encoding_early_fusion": rgb_models.SkeletonRgbEncodingEarlyFusion,

            # ------------------------------------------------------
            # ----------------    SKELETON + IMU    ----------------
            # ------------------------------------------------------
            "skeleton_imu_enhanced": SkeletonImuEnhancedModel,

            # ------------------------------------------------------
            # ---------------- SKELETON + RGB + IMU ----------------
            # ------------------------------------------------------
        }

        if mode not in modes:
            raise ValueError("Unsupported mode: " + mode)

        self._model = modes[mode](data_shape, num_classes, graph, **kwargs)

    def forward(self, x):
        return self._model(x)
