import torch.nn as nn

import models.mmagcn.agcn as agcn


# noinspection PyAbstractClass
class SkeletonImuEnhancedModel(nn.Module):
    """
    Take skeleton data with additional joints for each imu modality and append them to different parts of the skeleton.
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
