"""Spatial Configuration Network (SCN) from the paper
"Integrating spatial configuration into heatmap regression based CNNs for landmark localization"
by Payer et al. (2019). https://www.sciencedirect.com/science/article/pii/S1361841518305784"""

from typing import Sequence

import torch
from monai.networks.nets import UNet
from torch import nn


class SpatialConfigurationNet(nn.Module):
    """
    Adapted implementation of the Spatial Configuration Network (SCN) from the paper
    "Integrating spatial configuration into heatmap regression based CNNs for landmark localization"
    by Payer et al. (2019).
    https://www.sciencedirect.com/science/article/pii/S1361841518305784

    Args:
        spatial_dims (int): number of spatial dimensions of the input image.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        la_channels (Sequence[int], optional): number of output channels for each convolutional
            layer.
        la_kernel_size (int, optional): kernel size for the convolutional layers.
        la_strides (Sequence[int], optional): strides for the convolutional layers.
        la_num_res_units (int, optional): number of residual units in the convolutional layers.
        la_norm (str, optional): type of normalization to use. Defaults to "INSTANCE".
        la_activation (str, optional): type of activation to use. Defaults to "PRELU".
        la_adn_ordering (str, optional): ordering of the layers in the residual units. Defaults to
            "ADN".
        la_dropout (float, optional): dropout probability. Defaults to 0.0.
        sp_channels (int, optional): number of channels for the convolutional layers.
        sp_kernel_size (int, optional): kernel size for the convolutional layers.
        sp_downsample (int, optional): factor by which the image is downsampled.
        sp_image_input (bool, optional): whether to use the input image as input for the spatial
            configuration network.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 4,
        la_channels: Sequence[int] = (128, 128, 128, 128),
        la_kernel_size: int | tuple[int, int] = 3,
        # We use strides instead of max_pooling as in the paper as it is superior.
        # https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling
        la_strides: Sequence[int] = (2, 2, 2),
        la_num_res_units: int = 2,
        la_norm: str = "INSTANCE",
        la_activation: str = "PRELU",
        la_adn_ordering: str = "ADN",
        la_dropout: float = 0.0,  # In the paper, they use 0.5 dropout, however,
        # this seems to be too much, since we use residual connections.
        sp_channels: int = 128,
        sp_kernel_size: int = 11,
        sp_downsample: int = 16,
        sp_image_input: bool = True,
    ):
        super().__init__()
        self.sp_image_input = sp_image_input
        self.la_net = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=la_channels,
            strides=la_strides,
            num_res_units=la_num_res_units,
            norm=la_norm,
            dropout=la_dropout,
            act=la_activation,
            kernel_size=la_kernel_size,
            adn_ordering=la_adn_ordering,
        )
        self.sc_net = nn.Sequential(
            nn.AvgPool2d(kernel_size=sp_downsample),
            nn.Conv2d(
                in_channels=out_channels + int(self.sp_image_input) * in_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(
                in_channels=sp_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(
                in_channels=sp_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(
                in_channels=sp_channels,
                out_channels=out_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.Upsample(scale_factor=sp_downsample, mode="bicubic", align_corners=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_channels, *img_dims)
        """
        out_la = self.la_net(x)
        if self.sp_image_input:
            out_sc = self.sc_net(torch.cat((x, out_la), dim=1))
        else:
            out_sc = self.sc_net(out_la)
        out = out_la * out_sc
        return out


class OriginalSpatialConfigurationNet(nn.Module):
    """
    Implementation of the Spatial Configuration Network (SCN) from the paper
    "Integrating spatial configuration into heatmap regression based CNNs for landmark localization"
    by Payer et al. (2019).
    https://www.sciencedirect.com/science/article/pii/S1361841518305784

    Args:
        in_channels (int, optional): number of input channels. Defaults to 1.
        out_channels (int, optional): number of output channels. Defaults to 4.
        la_channels (int, optional): number of output channels for each convolutional layer.
            Defaults to 128.
        la_depth (int, optional): number of convolutional layers. Defaults to 3.
        la_kernel_size (int, optional): kernel size for the convolutional layers. Defaults to 3.
        la_dropout (float, optional): dropout probability. Defaults to 0.5.
        sp_channels (int, optional): number of channels for the convolutional layers. Defaults to
            128.
        sp_kernel_size (int, optional): kernel size for the convolutional layers. Defaults to 11.
        sp_downsample (int, optional): factor by which the image is downsampled. Defaults to 16.
        init_weights (bool, optional): whether to initialize the weights of the convolutional
            layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        la_channels: int = 128,
        la_depth: int = 3,
        la_kernel_size: int | tuple[int, ...] = 3,
        la_dropout: float = 0.5,
        sp_channels: int = 128,
        sp_kernel_size: int = 11,
        sp_downsample: int = 16,
        init_weigths: bool = False,
        spatial_dim: int = 2,
    ) -> None:
        super().__init__()
        self.init_weights = init_weigths
        if spatial_dim == 2:
            conv = nn.Conv2d  # type: ignore
            avg_pool = nn.AvgPool2d  # type: ignore
            mode = "bilinear"
        elif spatial_dim == 3:
            conv = nn.Conv3d  # type: ignore
            avg_pool = nn.AvgPool3d  # type: ignore
            mode = "trilinear"
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {spatial_dim}.")
        self.first_conv = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=la_channels,
                kernel_size=la_kernel_size,
                stride=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.la_downlayers = nn.ModuleList(
            [
                DownLayer(
                    in_channels=la_channels,
                    out_channels=la_channels,
                    dropout=la_dropout,
                    kernel_size=la_kernel_size,
                    spatial_dim=spatial_dim,
                )
                for _ in range(la_depth)
            ]
        )
        self.up_layers = nn.ModuleList(
            [
                UpLayer(
                    in_channels=la_channels,
                    out_channels=la_channels,
                    kernel_size=la_kernel_size,
                    spatial_dim=spatial_dim,
                )
                for _ in range(la_depth)
            ]
        )
        self.bottleneck_layer = nn.Sequential(
            conv(
                in_channels=la_channels,
                out_channels=la_channels,
                kernel_size=la_kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.1),
            conv(
                in_channels=la_channels,
                out_channels=la_channels,
                kernel_size=la_kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.last_layer = conv(
            in_channels=la_channels, out_channels=out_channels, kernel_size=1, padding="same"
        )

        self.sc_net = nn.Sequential(
            avg_pool(kernel_size=sp_downsample),
            conv(
                in_channels=out_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.1),
            conv(
                in_channels=sp_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.1),
            conv(
                in_channels=sp_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.1),
            conv(
                in_channels=sp_channels,
                out_channels=out_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.Upsample(scale_factor=sp_downsample, mode=mode, align_corners=False),
            nn.Tanh(),
        )
        if self.init_weights:
            self.apply(lambda m: self._init_weights(m, mean=0.0, std=0.0001))

    def _init_weights(self, module: nn.Module, mean: float = 0.0, std: float = 0.001) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.trunc_normal_(module.weight, mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_channels, *img_dims)
        """
        out_la = self.first_conv(x)
        skips = []
        for down_layer in self.la_downlayers:
            out_la, skip = down_layer(out_la)
            skips.append(skip)
        out_la = self.bottleneck_layer(out_la)
        for up_layer, skip in zip(self.up_layers, reversed(skips)):
            out_la = up_layer(out_la, skip)
        out_la = self.last_layer(out_la)
        out_sc = self.sc_net(out_la)
        out = out_la * out_sc
        return out


class DownLayer(nn.Module):
    """
    Down layer of the Localisation Network (LA) (UNet) from the original SCN paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        kernel_size: int | tuple[int, int] = 3,
        bias: bool = True,
        spatial_dim: int = 2,
    ) -> None:
        super().__init__()
        if spatial_dim == 2:
            conv = nn.Conv2d
            avg_pool = nn.AvgPool2d
            dropout_func = nn.Dropout2d
        elif spatial_dim == 3:
            conv = nn.Conv3d
            avg_pool = nn.AvgPool3d
            dropout_func = nn.Dropout3d
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {spatial_dim}.")
        self.conv1 = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=bias,
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.dropout = dropout_func(p=dropout)
        self.conv2 = nn.Sequential(
            conv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=bias,
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.pool = avg_pool(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output tensor of shape (batch_size, out_channels,
                *img_dims) and skip connection tensor of shape (batch_size, out_channels, *img_dims)
        """
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.pool(out), out


class UpLayer(nn.Module):
    """
    Up layer of the Localisation Network (LA) (UNet) from the original SCN paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        bias: bool = True,
        spatial_dim: int = 2,
    ):
        super().__init__()
        if spatial_dim == 2:
            conv = nn.Conv2d
            mode = "bilinear"
        elif spatial_dim == 3:
            conv = nn.Conv3d
            mode = "trilinear"
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {spatial_dim}.")
        self.conv1 = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=bias,
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)
            skip: skip connection tensor of shape (batch_size, out_channels, *img_dims)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_channels, *img_dims)
        """
        return self.upsample(x) + self.conv1(skip)


class OriginalSpatialConfigurationNet3d(OriginalSpatialConfigurationNet):
    """
    Implementation of the Spatial Configuration Network (SCN) from the paper
    "Integrating spatial configuration into heatmap regression based CNNs for landmark localization"
    by Payer et al. (2019).
    https://www.sciencedirect.com/science/article/pii/S1361841518305784

    This is the 3D version of the original SCN.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        la_channels: int = 64,
        la_depth: int = 3,
        la_kernel_size: int | tuple[int, ...] = 3,
        la_dropout: float = 0.5,
        sp_channels: int = 64,
        sp_kernel_size: int = 7,
        sp_downsample: int = 4,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            la_channels=la_channels,
            la_depth=la_depth,
            la_kernel_size=la_kernel_size,
            la_dropout=la_dropout,
            sp_channels=sp_channels,
            sp_kernel_size=sp_kernel_size,
            sp_downsample=sp_downsample,
            spatial_dim=3,
        )


class ProbSpatialConfigurationNet(nn.Module):
    """
    Probabilistic Spatial Configuration Network (PSCN)

    Adapted implementation of the Probabilistic Spatial Configuration Network (PSCN) from the paper
    "Integrating spatial configuration into heatmap regression based CNNs for landmark localization"
    by Payer et al. (2019). This is the same as the Spatial Configuration Network (SCN), but with
    a different last layer. Instead of multiplying the output of the SCN with the output of the
    spatial configuration network, we add them together, since the output of the spatial
    configuration network is a probability distribution in the logit space.

    Args:
        spatial_dims (int, optional): number of spatial dimensions of the input image. Defaults to
            2.
        in_channels (int, optional): number of input channels. Defaults to 1.
        out_channels (int, optional): number of output channels. Defaults to 4.
        la_channels (Sequence[int], optional): number of output channels for each convolutional
            layer. Defaults to (128, 128, 128, 128).
        la_kernel_size (int | tuple[int, int], optional): kernel size for the convolutional
            layers. Defaults to 3.
        la_strides (Sequence[int], optional): strides for the convolutional layers. Defaults to
            (2, 2, 2).
        la_num_res_units (int, optional): number of residual units in the convolutional layers.
            Defaults to 2.
        la_norm (str, optional): type of normalization to use. Defaults to "instance".
        la_activation (str, optional): type of activation to use. Defaults to "PRELU".
        la_adn_ordering (str, optional): ordering of the layers in the residual units. Defaults to
            "NDA".
        la_dropout (float, optional): dropout probability. Defaults to 0.0.
        sp_channels (int, optional): number of channels for the convolutional layers. Defaults to
            128.
        sp_kernel_size (int, optional): kernel size for the convolutional layers. Defaults to 11.
        sp_downsample (int, optional): factor by which the image is downsampled. Defaults to 16.
        sp_image_input (bool, optional): whether to use the input image as input for the spatial
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 4,
        la_channels: Sequence[int] = (128, 128, 128, 128, 128),
        la_kernel_size: int | tuple[int, int] = 3,
        # We use strides instead of max_pooling as in the paper as it is superior.
        # https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling
        la_strides: Sequence[int] = (2, 2, 2, 2),
        la_num_res_units: int = 2,
        la_norm: str = "instance",
        la_activation: str = "PRELU",
        la_adn_ordering: str = "NDA",
        la_dropout: float = 0.0,  # In the paper, they use 0.5 dropout, however,
        # this seems to be too much, since we use residual connections.
        sp_channels: int = 128,
        sp_kernel_size: int = 11,
        sp_downsample: int = 16,
        sp_image_input: int = True,
    ) -> None:
        super().__init__()
        self.sp_image_input = sp_image_input
        self.la_net = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=la_channels,
            strides=la_strides,
            num_res_units=la_num_res_units,
            norm=la_norm,
            dropout=la_dropout,
            act=la_activation,
            kernel_size=la_kernel_size,
            adn_ordering=la_adn_ordering,
        )
        self.sc_net = nn.Sequential(
            nn.AvgPool2d(kernel_size=sp_downsample),
            nn.Conv2d(
                in_channels=out_channels + int(self.sp_image_input) * in_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=sp_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=sp_channels,
                out_channels=sp_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=sp_channels,
                out_channels=out_channels,
                kernel_size=sp_kernel_size,
                padding="same",
            ),
            nn.Upsample(scale_factor=sp_downsample, mode="bicubic", align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, *img_dims)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_channels, *img_dims)
        """
        out_la = self.la_net(x)
        if self.sp_image_input:
            out_sc = self.sc_net(torch.cat((x, out_la), dim=1))
        else:
            out_sc = self.sc_net(out_la)
        out = out_la + out_sc
        return out
