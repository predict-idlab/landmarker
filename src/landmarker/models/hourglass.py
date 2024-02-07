"""Hourglass network.
Proposed in: https://github.com/princeton-vl/pytorch_stacked_hourglass/tree/master
Partially inspired by: "Stacked Hourglass Network for Robust Facial Landmark Localisation"
            - Yang et al. (2017)
"""

from typing import Sequence

import torch
from monai.networks.blocks import Convolution, ResidualUnit
from torch import nn


class Hourglass(nn.Module):
    """
    Hourglass network is a network with symmetrical encoder and decoder paths. The encoder path
    downsamples the input image while the decoder path upsamples the image. Skip connections are
    added between the encoder and decoder paths to preserve spatial information. This network is
    used for pose estimation.
        Proposed in: "Stacked Hourglass Networks for Human Pose Estimation" - Newell et al. (2016)

    Args:
        spatial_dims (int): number of spatial dimensions of the input image.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        channels (Sequence[int], optional): number of output channels for each convolutional layer.
        conv_block (nn.Module, optional): convolutional block to use. Defaults to ResidualUnit.
        pooling (nn.Module, optional): pooling layer to use. Defaults to nn.MaxPool2d.
        up_sample_mode (str, optional): upsampling mode. Defaults to 'nearest'.
    """

    # TODO: implment the out_channels

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = [64, 128, 256, 512],
        conv_block: nn.Module = ResidualUnit,  # type: ignore[assignment]
        pooling: nn.Module = nn.MaxPool2d,  # type: ignore[assignment]
        up_sample_mode: str = "nearest",
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.depth = len(channels)
        self.down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_block(
                        spatial_dims=self.spatial_dims,
                        in_channels=in_channels,
                        out_channels=channels[0],
                    ),
                    pooling(2),
                )
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_block(
                        spatial_dims=self.spatial_dims,
                        in_channels=channels[-1],
                        out_channels=channels[-1],
                    ),
                    nn.Upsample(scale_factor=2, mode=up_sample_mode),
                )
            ]
        )
        self.skip_blocks = nn.ModuleList(
            [
                conv_block(
                    spatial_dims=self.spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[-1],
                )
            ]
        )

        for i, channel in enumerate(channels[1:]):
            self.down_blocks.append(
                nn.Sequential(
                    conv_block(
                        spatial_dims=self.spatial_dims,
                        in_channels=channels[i],
                        out_channels=channel,
                    ),
                    pooling(2),
                )
            )
            self.skip_blocks.append(
                conv_block(
                    spatial_dims=self.spatial_dims, in_channels=channel, out_channels=channels[-1]
                )
            )
            self.up_blocks.append(
                nn.Sequential(
                    conv_block(
                        spatial_dims=self.spatial_dims,
                        in_channels=channels[-1],
                        out_channels=channels[-1],
                    ),
                    nn.Upsample(scale_factor=2, mode=up_sample_mode),
                )
            )
        self.neck_block = nn.Sequential(
            conv_block(
                spatial_dims=self.spatial_dims, in_channels=channels[-1], out_channels=channels[-1]
            ),
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=channels[-1],
                out_channels=channels[-1],
                kernel_size=1,
                adn_ordering="N",
                norm="batch",
            ),  # TODO: check if this is correct (if it makes sense)
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=channels[-1],
                out_channels=channels[-1],
                kernel_size=1,
                adn_ordering="N",
                norm="batch",
            ),  # TODO: check if this is correct (if it makes sense)
        )
        self.last_layer = nn.Sequential(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=channels[-1],
                out_channels=channels[-1] // 2,
                kernel_size=1,
                adn_ordering="N",
                norm="batch",
            ),
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=channels[-1] // 2,
                out_channels=out_channels,
                kernel_size=1,
                adn_ordering="",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)
        Returns:
            output tensor of shape (batch_size, out_channels, *img_dims)
        """
        down_outputs = []
        for down_block, skip_block in zip(self.down_blocks, self.skip_blocks):
            x = down_block(x)
            down_outputs.append(skip_block(x))
        x = self.neck_block(x)
        for up_block, skip_output in zip(self.up_blocks[::-1], down_outputs[::-1]):
            x = up_block(x + skip_output)
        x = self.last_layer(x)
        return x


class StackedHourglass(nn.Module):
    """
    Stacked hourglass.

    Args:
        nb_stacks (int): number of hourglass modules to stack.
        spatial_dims (int): number of spatial dimensions of the input image.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        channels (Sequence[int], optional): number of output channels for each convolutional layer.

        up_sample_mode (str, optional): upsampling mode. Defaults to 'nearest'.
    """

    def __init__(
        self,
        nb_stacks: int,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = [64, 128, 256, 512],
        conv_block: nn.Module = ResidualUnit,  # type: ignore[assignment]
        pooling: nn.Module = nn.MaxPool2d,  # type: ignore[assignment]
        up_sample_mode: str = "nearest",
    ):
        super().__init__()
        self.nb_stacks = nb_stacks
        # TODO: in the orignal paper they add a pre sequential block
        self.stacks = nn.ModuleList(
            [
                Hourglass(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    channels=channels,
                    conv_block=conv_block,
                    pooling=pooling,
                    up_sample_mode=up_sample_mode,
                )
                for _ in range(nb_stacks)
            ]
        )
        self.output_to_feature_block = nn.ModuleList(
            [
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    adn_ordering="N",
                    norm="batch",
                )
                for _ in range(nb_stacks - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)
        Returns:
            list of output tensors of shape (batch_size, out_channels, *img_dims)
        """
        outputs = []
        next_x = x
        for i, stack in enumerate(self.stacks):
            pred = stack(next_x)
            outputs.append(pred)
            if i < self.nb_stacks - 1:
                next_x = x + self.output_to_feature_block[i](pred)
        return outputs
