"""CoordConv implementation proposed in "An intriguing failing of convolutional neural networks and
    the CoordConv solution" - Liu et al."""

import torch
from torch import nn
from monai.networks.blocks import ResidualUnit


class CoordConvLayer(nn.Module):
    """
    CoordConv is a convolutional layer that adds the x and y coordinates of each pixel as additional
    channels to the input tensor. Optionally, it can also add the radial distance of each pixel to
    the center of the image as an additional channel. This is done to provide the network with
    spatial information.
        source: "An intriguing failing of convolutional neural networks and the CoordConv
            solution" - Liu et al.

    Args:
        spatial_dims (int): number of spatial dimensions of the input image.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        radial_channel (bool, optional): whether to add the radial distance of each pixel to the
            center of the image as an additional channel. Defaults to False.
        conv_block (nn.Module, optional): convolutional block to use. Defaults to ResidualUnit.
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int,
                 radial_channel: bool = False,
                 conv_block: nn.Module = ResidualUnit  # type: ignore[assignment]
                 ):
        super().__init__()
        self.radial_channel = radial_channel
        self.add_coord_channels = AddCoordChannels(radial_channel=radial_channel)
        self.conv_block = conv_block(spatial_dims=spatial_dims,
                                     in_channels=in_channels + 2 + int(radial_channel),
                                     out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *spatial_dims)

        Returns:
            output tensor of shape (batch_size, out_channels, *spatial_dims)
        """
        out = self.add_coord_channels(x)
        out = self.conv_block(out)
        return out


class AddCoordChannels(nn.Module):
    """
    Adds the x and y coordinates of each pixel as additional channels to the input tensor.
    Optionally, it can also add the radial distance of each pixel to the center of the image as an
    additional channel. This is done to provide the network with spatial information.

    Args:
        radial_channel (bool, optional): whether to add the radial distance of each pixel to the
            center of the image as an additional channel. Defaults to False.
    """

    def __init__(self, radial_channel: bool = False) -> None:
        super().__init__()
        self.radial_channel = radial_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *spatial_dims)

        Returns:
            output tensor of shape (batch_size, in_channels + 2 + radial_channel, *spatial_dims)
        """
        b, _, h, w = x.shape
        xs = torch.arange(0, w, dtype=torch.float32, requires_grad=True).to(x)
        ys = torch.arange(0, h, dtype=torch.float32, requires_grad=True).to(x)
        xs, ys = torch.meshgrid(xs, ys, indexing='xy')
        xs = xs.unsqueeze(0).unsqueeze(0)
        ys = ys.unsqueeze(0).unsqueeze(0)
        xs = xs / (w - 1)
        ys = ys / (h - 1)
        xs = xs * 2 - 1
        ys = ys * 2 - 1
        xs = xs.repeat(b, 1, 1, 1)
        ys = ys.repeat(b, 1, 1, 1)
        if self.radial_channel:
            r = torch.sqrt(torch.pow(xs, 2) + torch.pow(ys, 2))
            return torch.cat((x, ys, xs, r), 1)
        return torch.cat((x, ys, xs), 1)
