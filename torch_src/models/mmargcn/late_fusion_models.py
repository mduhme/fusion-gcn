import torch
import torch.nn as nn

import models.mmargcn.agcn as agcn
import models.mmargcn.imu_feature_models as imu_models
import models.mmargcn.rgb_feature_models as rgb_models
import models.mmargcn.early_fusion_models as early_fusion_models
from models.mmargcn.fusion import get_fusion, get_skeleton_imu_fusion_graph


class SkeletonRgbR2P1D(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)
        dropout = kwargs.get("dropout", 0.)
        fusion_type = kwargs.get("fusion", "concatenate")
        self.r2p1d = rgb_models.RgbR2p1DModel(data_shape["rgb"], num_classes, graph, without_fc=True, model_depth=18,
                                              **kwargs)
        self.agcn = agcn.Model(data_shape["skeleton"], num_classes, graph, num_layers=num_layers, without_fc=True,
                               dropout=dropout)
        self.fusion = get_fusion(fusion_type, concatenate_dim=-1)
        self.fc1 = nn.Linear(self.r2p1d.r2p1d.out_dim, self.agcn.out_channels)

        if fusion_type == "concatenate":
            out_dim = self.agcn.out_channels * 2
        else:
            out_dim = self.agcn.out_channels

        self.fc2 = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        skeleton_data = x["skeleton"]
        rgb_data = x["rgb"]

        skeleton_data = self.agcn(skeleton_data)
        rgb_data = self.r2p1d(rgb_data)
        rgb_data = self.fc1(rgb_data)

        fused_data = self.fusion.combine(skeleton_data, rgb_data)

        y = self.fc2(fused_data)
        return y


class SkeletonImuGCN(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)
        dropout = kwargs.get("dropout", 0.)
        fusion_type = kwargs.get("fusion", "concatenate")
        if kwargs.pop("skeleton_imu_enhanced", False):
            graph = get_skeleton_imu_fusion_graph(graph, **kwargs)
        self.imu_gcn = imu_models.ImuGCN(data_shape, num_classes, inter_signal_back_connections=True,
                                         include_additional_top_layer=True, without_fc=True, **kwargs)
        self.agcn = agcn.Model(data_shape["skeleton"], num_classes, graph, num_layers=num_layers, without_fc=True,
                               dropout=dropout)
        self.fusion = get_fusion(fusion_type, concatenate_dim=-1)

        if fusion_type == "concatenate":
            out_dim = self.agcn.out_channels * 2
        else:
            out_dim = self.agcn.out_channels

        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        skeleton_data = x["skeleton"]
        inertial_data = x["inertial"]

        skeleton_data = self.agcn(skeleton_data)
        inertial_data = self.imu_gcn(inertial_data)

        fused_data = self.fusion.combine(skeleton_data, inertial_data)
        y = self.fc(fused_data)
        return y

