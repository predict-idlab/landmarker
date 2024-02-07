"""Stacked Cholesky Hourglass Network as proposed in "UGLLI Face Alignment: Estimating Uncertainty
with with Gaussian Log-Likelihood Loss" - Kumar et al. (2019)

#TODO: Check if the implementation is correct. The LUVLi paper seems to provide some code.
# see: https://github.com/abhi1kumar/LUVLi
"""

from typing import Sequence

import torch
from monai.networks.blocks import Convolution, ResidualUnit
from torch import nn

from landmarker.models.hourglass import Hourglass


class StackedCholeskyHourglass(nn.Module):
    """
    Stacked Cholesky Hourglass Network as proposed in "UGLLI Face Alignment: Estimating Uncertainty
    with Gaussian Log-Likelihood Loss" - Kumar et al. (2019).  It is a stack of hourglass networks
    with a Cholesky Estimator Network at the bottleneck of each hourglass. The output of the
    Cholesky Estimator Network is a lower triangular matrix that is used to estimate the covariance
    matrix of the Gaussian distribution of the predicted heatmaps. The covariance matrix is then
    used to compute the Gaussian Log-Likelihood Loss.

    Args:
        nb_stacks (int): number of hourglass networks to stack.
        img_size (tuple[int, int]): size of the input image.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        channels (Sequence[int], optional): number of output channels for each convolutional layer.
        conv_block (nn.Module, optional): convolutional block to use. Defaults to ResidualUnit.
        up_sample_mode (str, optional): upsampling mode. Defaults to 'nearest'.
    """

    def __init__(
        self,
        nb_stacks: int,
        img_size: tuple[int, int],
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = [64, 128, 256, 512],
        conv_block: nn.Module = ResidualUnit,  # type: ignore[assignment]
        up_sample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.nb_stacks = nb_stacks
        self.stacks = nn.ModuleList(
            [
                CholeskyHourglass(
                    img_size=img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    channels=channels,
                    conv_block=conv_block,
                    up_sample_mode=up_sample_mode,
                )
                for _ in range(nb_stacks)
            ]
        )
        self.output_to_feature_block = nn.ModuleList(
            [
                Convolution(
                    spatial_dims=len(img_size),
                    in_channels=out_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    adn_ordering="N",
                    norm="batch",
                )
                for _ in range(nb_stacks - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)

        Returns:
            heatmaps: list of predicted heatmaps of shape (batch_size, out_channels, *img_dims)
            cens: list of covariance matrices of the predicted heatmaps of shape (batch_size,
                out_channels, 2, 2)
        """
        heatmaps, cens = [], []
        next_x = x
        for i, stack in enumerate(self.stacks):
            pred, output_cen = stack(next_x)
            cens.append(output_cen)
            heatmaps.append(pred)
            if i < self.nb_stacks - 1:
                next_x = x + self.output_to_feature_block[i](pred)
        return heatmaps, cens


class CholeskyHourglass(nn.Module):
    """
    Proposed in "UGLLI Face Alignment: Estimating Uncertainty with
    Gaussian Log-Likelihood Loss" - Kumar et al. (2019)
    # TODO: Note that the implementation of Kumar et al. use DU-Net as the backbone.
    # We use the residual hourglass.

    Args:
        img_size (tuple[int, int]): size of the input image.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        channels (Sequence[int], optional): number of output channels for each convolutional layer.
        subunits (int, optional): number of subunits in each convolutional layer.
        up_sample_mode (str, optional): upsampling mode. Defaults to 'nearest'.

    Returns:
        pred: predicted heatmaps of shape (batch_size, out_channels, *img_dims)
        cen: covariance matrices of the predicted heatmaps of shape (batch_size,
            out_channels, 2, 2)
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = [64, 128, 256, 512],
        conv_block: nn.Module = ResidualUnit,  # type: ignore[assignment]
        up_sample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.hg = Hourglass(
            spatial_dims=len(img_size),
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            conv_block=conv_block,
            up_sample_mode=up_sample_mode,
        )
        self.img_size = img_size
        self.cen = CholeskyBlock(
            nb_classes=out_channels,
            input_units=channels[-1] * (img_size[0] // (2 ** len(channels))) ** 2,
            hidden_units=128,
            class_output_units=3,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)

        Returns:
            pred: predicted heatmaps of shape (batch_size, out_channels, *img_dims)
            cen: covariance matrices of the predicted heatmaps of shape (batch_size,
                out_channels, 2, 2)
        """
        down_outputs = []
        for down_block, skip_block in zip(self.hg.down_blocks, self.hg.skip_blocks):
            x = down_block(x)
            down_outputs.append(skip_block(x))
        x = self.hg.neck_block(x)
        cen_output = self.cen(x.flatten(1))
        for up_block, skip_output in zip(self.hg.up_blocks[::-1], down_outputs[::-1]):
            x = up_block(x + skip_output)
        x = self.hg.last_layer(x)
        return x, cen_output


class CholeskyBlock(nn.Module):
    """
    Cholesky Estimator Network tries to estimate the Cholesky decomposition of the covariance matrix
    out of the bottleneck of each hourglass of the stack.

    Proposed in "UGLLI Face Alignment: Estimating Uncertainty with
    Gaussian Log-Likelihood Loss" - Kumar et al. (2019)

    Args:
        nb_classes (int): number of output classes.
        input_units (int): number of input units.
        hidden_units (int): number of hidden units.
        class_output_units (int): number of output units.
        elu_alpha (float, optional): alpha parameter of the ELU activation function.
            Defaults to 1.0.
    """

    def __init__(
        self,
        nb_classes: int,
        input_units: int,
        hidden_units: int,
        class_output_units: int,
        elu_alpha: float = 1.0,
    ):
        super().__init__()
        self.nb_classes = nb_classes
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.class_output_units = class_output_units
        self.seq = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, nb_classes * class_output_units),
        )
        self.elu_alpha = elu_alpha
        self.elu = nn.ELU(alpha=elu_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, input_units)

        Returns:
            cov: covariance matrices of the predicted heatmaps of shape (batch_size,
                out_channels, 2, 2)
        """
        cholesky_decom = self.seq(x).view(-1, self.nb_classes, self.class_output_units)
        # Transform the output of the network to a lower triangular matrix
        l_mat = torch.zeros(
            (cholesky_decom.shape[0], self.nb_classes, 2, 2), device=cholesky_decom.device
        )
        l_mat[:, :, 0, 0] = self.elu(cholesky_decom[:, :, 0]) + self.elu_alpha
        l_mat[:, :, 1, 0] = cholesky_decom[:, :, 1]
        l_mat[:, :, 1, 1] = self.elu(cholesky_decom[:, :, 2]) + self.elu_alpha
        cov = l_mat @ l_mat.transpose(-2, -1)
        return cov
