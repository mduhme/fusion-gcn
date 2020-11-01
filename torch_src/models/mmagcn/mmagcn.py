import math
import torch
import torch.nn as nn

import models.mmagcn.agcn as agcn
import models.mmagcn.resnet2p1d as r2p1d


# noinspection PyAbstractClass
class SkeletonImuEnhanced(nn.Module):
    """
    Take skeleton data with additional joints for each imu modality and append them to the center joint (shoulders)
    """
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)
        imu_enhanced_mode = kwargs["imu_enhanced_mode"]
        num_imu_joints = kwargs["num_imu_joints"]

        # create new edges for imu data
        new_edges = []
        if imu_enhanced_mode == "append_center":
            # append imu joints to skeleton center joint
            center_joint = kwargs.get("center_joint", graph.center_joint)
            new_edges.extend((graph.num_vertices + i, center_joint) for i in range(num_imu_joints))

        elif imu_enhanced_mode == "append_right":
            # append imu joints to skeleton right wrist and right hip
            right_wrist_joint = kwargs["right_wrist_joint"]
            right_hip_joint = kwargs["right_hip_joint"]
            for i in range(num_imu_joints):
                new_edges.append((graph.num_vertices + i, right_wrist_joint))
                new_edges.append((graph.num_vertices + i, right_hip_joint))

        else:
            raise ValueError("Unsupported imu_enhanced_mode: " + imu_enhanced_mode)

        if kwargs.get("interconnect_imu_joints", False):
            for i in range(num_imu_joints):
                for j in range(i + 1, num_imu_joints):
                    new_edges.append((graph.num_vertices + i, graph.num_vertices + j))

        skeleton_imu_graph = graph.with_new_edges(new_edges)
        self._agcn = agcn.Model(data_shape["skeleton"], num_classes, skeleton_imu_graph,
                                num_layers=num_layers, without_fc=False)

    def forward(self, x):
        return self._agcn(x)


# noinspection PyAbstractClass
class SkeletonImuCombinedRgbPatches(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)

        # create new edges for imu data, append them to center joint
        skeleton_imu_graph = graph.with_new_edges([
            (graph.num_vertices, graph.center_joint),
            (graph.num_vertices + 1, graph.center_joint)
        ])
        # self._skeleton_model = agcn.Model(data_shape["skeleton"], num_classes, skeleton_imu_graph,
        #                                   num_layers=num_layers, without_fc=True)
        self._rgb_model = agcn.Model(data_shape["rgb"], num_classes, graph,
                                     num_layers=num_layers, start_feature_size=512, without_fc=False)
        # self.fc = nn.Linear(self._skeleton_model.out_channels + self._rgb_model.out_channels, num_classes)
        # nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))

    def forward(self, x):
        skeleton_x, rgb_x = x
        # skeleton_x = self._skeleton_model(skeleton_x)
        rgb_x = self._rgb_model(rgb_x)
        # x = torch.cat((skeleton_x, rgb_x), 1)
        # x = self.fc(x)
        return rgb_x


# noinspection PyAbstractClass
class Model(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, mode: str, **kwargs):
        super().__init__()
        self._model = None
        if mode == "skeleton_imu_enhanced":
            self._model = SkeletonImuEnhanced(data_shape, num_classes, graph, **kwargs)
        elif mode == "skele+imu__rgb_patches_op":
            self._model = SkeletonImuCombinedRgbPatches(data_shape, num_classes, graph, **kwargs)

    def forward(self, x):
        return self._model(x)
