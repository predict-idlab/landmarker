import torch
from torch import nn


class SoftmaxND(nn.Module):
    def __init__(self, spatial_dims):
        super().__init__()
        self.dim = (2, 3) if spatial_dims == 2 else (2, 3, 4)

    def forward(self, x):
        out = torch.exp(x)
        return out / torch.sum(out, dim=self.dim, keepdim=True)
