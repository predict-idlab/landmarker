import torch
from torch import nn


class SoftmaxND(nn.Module):
    def __init__(self, spatial_dims):
        super().__init__()
        self.dim = (-2, -1) if spatial_dims == 2 else (-3, -2, -1)

    def forward(self, x):
        max_val = x
        for d in self.dim:
            max_val, _ = torch.max(max_val, dim=d, keepdim=True)
        out = torch.exp(x - max_val)
        return out / torch.sum(out, dim=self.dim, keepdim=True)


class LogSoftmaxND(nn.Module):
    def __init__(self, spatial_dims):
        super().__init__()
        self.dim = (-2, -1) if spatial_dims == 2 else (-3, -2, -1)
        self.spatial_dims = spatial_dims

    def forward(self, x):
        if self.spatial_dims == 2:
            out_max, _ = torch.max(x, dim=-1, keepdim=True)
            out_max, _ = torch.max(out_max, dim=-2, keepdim=True)
        else:
            out_max, _ = torch.max(x, dim=-1, keepdim=True)
            out_max, _ = torch.max(out_max, dim=-2, keepdim=True)
            out_max, _ = torch.max(out_max, dim=-3, keepdim=True)
        x_exp = torch.exp(x - out_max)
        x_exp_sum = torch.sum(x_exp, dim=self.dim, keepdim=True)
        return x - out_max - torch.log(x_exp_sum)
