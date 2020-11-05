import torch.nn as nn

from models.mmagcn.gcn import GCN


class ImuGCN(nn.Module):
    def __init__(self, data_shape, num_classes: int, **kwargs):
        super().__init__()
        dropout = kwargs.get("dropout", 0.)
        self.gcn = GCN(None, 0, 0, num_classes, dropout)

    def forward(self, x) -> None:
        return self.gcn(x)
