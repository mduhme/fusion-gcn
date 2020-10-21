import math
import torch.nn as nn

import models.mmagcn.agcn as agcn


# noinspection PyAbstractClass
class SkeletonImuCombined(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)

        # create new edges for imu data, append them to center joint
        skeleton_imu_graph = graph.with_new_edges([
            (graph.num_vertices, graph.center_joint),
            (graph.num_vertices + 1, graph.center_joint)
        ])
        self._agcn = agcn.Model(data_shape["skeleton"], num_classes, skeleton_imu_graph,
                                num_layers=num_layers, without_fc=False)

    def forward(self, x):
        return self._agcn(x)


# noinspection PyAbstractClass
class SkeletonImuCombinedRgbPatches(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 6)

        # create new edges for imu data, append them to center joint
        skeleton_imu_graph = graph.with_new_edges([
            (graph.num_vertices, graph.center_joint),
            (graph.num_vertices + 1, graph.center_joint)
        ])
        self._skeleton_model = agcn.Model(data_shape["skeleton"], num_classes, skeleton_imu_graph,
                                          num_layers=10, without_fc=False)
        # self._rgb_model = agcn.Model(data_shape["rgb"], num_classes, graph, num_layers=num_layers,
        #                              start_feature_size=512, without_fc=True)
        # self._fc = nn.Linear(self.layers[-1].out_channels, num_classes)
        # nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))

    def forward(self, x):
        skeleton_x, rgb_x = x
        skeleton_x = self._skeleton_model(skeleton_x)
        # rgb_x = self._rgb_model(rgb_x)
        return skeleton_x


# noinspection PyAbstractClass
class Model(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, mode: str, **kwargs):
        super().__init__()
        self._model = None
        if mode == "skele+imu":
            self._model = SkeletonImuCombined(data_shape, num_classes, graph, **kwargs)
        elif mode == "skele+imu__rgb_patches_op":
            self._model = SkeletonImuCombinedRgbPatches(data_shape, num_classes, graph, **kwargs)

    def forward(self, x):
        return self._model(x)
