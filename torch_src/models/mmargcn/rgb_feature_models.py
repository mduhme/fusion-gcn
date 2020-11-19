import os

import torch
import torch.nn as nn
import torchvision.models as models

import models.mmargcn.agcn as agcn
import models.mmargcn.resnet2p1d as r2p1d
from util.graph import Graph


class RgbPatchFeaturesModel(nn.Module):
    """
    Input data comes in form of precomputed features of extracted RGB patches for each frame.
    These features are treated like skeleton data, meaning there is one feature input for each of the skeleton joints
    (instead of R^3 = (x, y, z) skeleton feature now a R^512 feature computed by Resnet).
    """

    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)
        self.agcn = agcn.Model(data_shape["rgb"], num_classes, graph, num_layers=num_layers,
                               without_fc=kwargs.get("without_fc", False))

    def forward(self, x):
        return self.agcn(x)


# noinspection PyUnusedLocal
class RgbPatchGroupsFeaturesModel(nn.Module):
    """
    Take skeleton data with additional joints for each imu modality and append them to different parts of the skeleton.
    """

    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)

        edges = kwargs["rgb_patch_groups_edges"]
        edges = [tuple(map(int, edge.split(", "))) for edge in edges]
        graph = Graph(edges)

        self.agcn = agcn.Model(data_shape["rgb"], num_classes, graph, num_layers=num_layers,
                               without_fc=kwargs.get("without_fc", False))

    def forward(self, x):
        return self.agcn(x)


class RgbCnnEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_bodies = kwargs.get("rgb_num_bodies", 1)
        self.num_vertices = kwargs.get("rgb_num_vertices", 20)
        self.num_encoded_channels = kwargs.get("rgb_node_encoding_feature_dim", 3)

        torch_hub = os.path.abspath(kwargs.get("torch_hub", "../torchhome/hub"))
        torch.hub.set_dir(torch_hub)
        cnn = models.resnet18(pretrained=True)
        fc_input_size = 512

        layers_without_fc = list(cnn.children())[:-1]
        rgb_node_encoding_feature_dim = self.num_bodies * self.num_vertices * self.num_encoded_channels
        self.cnn = torch.nn.Sequential(*layers_without_fc)
        self.cnn_fc = torch.nn.Linear(fc_input_size, rgb_node_encoding_feature_dim)

    def forward(self, x):
        n, num_frames, num_channels, h, w = x.size()
        x = x.view(n * num_frames, num_channels, h, w)

        y = self.cnn(x)
        y = torch.squeeze(y)
        y = self.cnn_fc(y)
        y = y.view(n, num_frames, self.num_bodies, self.num_vertices, self.num_encoded_channels)
        y = y.permute(0, 2, 1, 3, 4).contiguous()

        return y


class RgbCnnEncoderModel(nn.Module):
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        num_layers = kwargs.get("num_layers", 10)

        self.rgb_encoder = RgbCnnEncoder(rgb_num_vertices=graph.num_vertices, **kwargs)

        agcn_input_shape = (self.rgb_encoder.num_bodies, data_shape["rgb"][0], self.rgb_encoder.num_vertices,
                            self.rgb_encoder.num_encoded_channels)
        self.agcn = agcn.Model(agcn_input_shape, num_classes, graph, num_layers=num_layers,
                               without_fc=kwargs.get("without_fc", False))

    def forward(self, x):
        x = self.rgb_encoder(x)
        x = self.agcn(x)
        return x


class RgbR2p1DModel(nn.Module):
    # noinspection PyUnusedLocal
    def __init__(self, data_shape, num_classes: int, graph, **kwargs):
        super().__init__()
        model_depth = kwargs.get("model_depth", 18)
        pretrained_weights_path = kwargs.get("pretrained_weights_path", None)
        self.r2p1d = r2p1d.generate_model(model_depth, pretrained_weights_path=pretrained_weights_path)
        if kwargs.get("without_fc", False):
            self.fc = lambda x: x
        else:
            self.fc = nn.Linear(self.r2p1d.out_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.r2p1d(x)
        x = self.fc(x)
        return x


class RgbR2P1DEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_encoded_channels = kwargs.get("rgb_node_encoding_feature_dim", 3)
        self.num_additional_nodes = kwargs.get("num_additional_nodes", 3)
        model_depth = kwargs.get("model_depth", 10)
        self.r2p1d = r2p1d.generate_model(model_depth, temporal_stride=1, no_avg=True)
        self.cnn = nn.Conv2d(self.r2p1d.out_dim, self.num_encoded_channels, kernel_size=(5, 1), padding=(2, 0))
        self.avgpool = nn.AdaptiveAvgPool2d((None, self.num_additional_nodes))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.r2p1d(x)
        x = torch.flatten(x, start_dim=3)
        x = self.cnn(x)
        x = self.avgpool(x)
        return x
