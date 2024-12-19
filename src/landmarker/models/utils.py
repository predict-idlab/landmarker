import torch
from torch import nn


class SoftmaxND(nn.Module):
    """
    Applies the Softmax function over N-dimensional input tensor.

    Args:
        spatial_dims (int): The number of spatial dimensions (2 or 3).

    Attributes:
        dim (tuple): The dimensions over which to apply the Softmax function.

    Methods:
        forward(x):
            Applies the Softmax function to the input tensor `x`.

    Example:
        >>> softmax_nd = SoftmaxND(spatial_dims=2)
        >>> input_tensor = torch.randn(1, 3, 4, 4)
        >>> output_tensor = softmax_nd(input_tensor)
    """

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
    """
    Applies the LogSoftmax function over N-dimensional input.

    Args:
        spatial_dims (int): The number of spatial dimensions (2 or 3).

    Attributes:
        dim (tuple): The dimensions over which to apply the LogSoftmax function.
        spatial_dims (int): The number of spatial dimensions.

    Methods:
        forward(x):
            Applies the LogSoftmax function to the input tensor `x`.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The tensor after applying the LogSoftmax function.
    """

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
