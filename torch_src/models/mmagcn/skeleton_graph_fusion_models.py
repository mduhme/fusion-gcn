import torch.nn as nn

import models.mmagcn.agcn as agcn
from models.mmagcn.fusion import get_fusion, get_skeleton_imu_fusion_graph
from models.mmagcn.rgb_feature_models import RgbEncoder


class SkeletonImuEnhancedModel(nn.Module):
    """
    Take skeleton data with additional joints for each imu modality and append them to different parts of the skeleton.
    """

    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)
        skeleton_imu_graph = get_skeleton_imu_fusion_graph(graph, **kwargs)
        self.agcn = agcn.Model(data_shape["skeleton"], num_classes, skeleton_imu_graph, num_layers=num_layers)

    def forward(self, x):
        return self.agcn(x)


class SkeletonRgbEarlyFusion(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)
        self._fusion_type = kwargs.get("fusion", "concatenate")

        self.rgb_encoder = RgbEncoder(rgb_num_vertices=graph.num_vertices, **kwargs)

        if self._fusion_type == "concatenate":
            num_channels = data_shape["skeleton"][-1] + self.rgb_encoder.num_encoded_channels
        else:
            num_channels = data_shape["skeleton"][-1]

        agcn_input_shape = (self.rgb_encoder.num_bodies, data_shape["rgb"][0], graph.num_vertices,
                            num_channels)
        self.agcn = agcn.Model(agcn_input_shape, num_classes, graph, num_layers=num_layers)
        self._fusion = get_fusion(self._fusion_type, concatenate_dim=-1)

    def forward(self, x):
        skeleton_data = x["skeleton"]
        rgb_data = x["rgb"]

        # Encode RGB images
        rgb_data = self.rgb_encoder(rgb_data)

        # Early fusion of skeleton and rgb
        fused_data = self._fusion.combine(skeleton_data, rgb_data)

        # Run graph convolutional neural network
        y = self.agcn(fused_data)
        return y


class SkeletonImuRgbEarlyFusion(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)
        skeleton_imu_graph = get_skeleton_imu_fusion_graph(graph, **kwargs)
        self._fusion_type = kwargs.get("fusion", "concatenate")

        self.rgb_encoder = RgbEncoder(rgb_num_vertices=skeleton_imu_graph.num_vertices, **kwargs)

        if self._fusion_type == "concatenate":
            num_channels = data_shape["skeleton"][-1] + self.rgb_encoder.num_encoded_channels
        else:
            num_channels = data_shape["skeleton"][-1]

        agcn_input_shape = (self.rgb_encoder.num_bodies, data_shape["rgb"][0], skeleton_imu_graph.num_vertices,
                            num_channels)
        self.agcn = agcn.Model(agcn_input_shape, num_classes, skeleton_imu_graph, num_layers=num_layers)
        self._fusion = get_fusion(self._fusion_type, concatenate_dim=-1)

    def forward(self, x):
        skeleton_data = x["skeleton"]
        rgb_data = x["rgb"]

        # Encode RGB images
        rgb_data = self.rgb_encoder(rgb_data)

        # Early fusion of skeleton and rgb
        fused_data = self._fusion.combine(skeleton_data, rgb_data)

        # Run graph convolutional neural network
        y = self.agcn(fused_data)
        return y
