"""Call get_model(model_name, **kwargs) to get a model."""

import os
from typing import Optional

import torch
from monai.networks.nets import UNet
from torch import nn

from .cholesky_hourglass import CholeskyHourglass, StackedCholeskyHourglass
from .coord_conv import AddCoordChannels, CoordConvLayer
from .hourglass import Hourglass, StackedHourglass
from .spatial_configuration_net import (
    OriginalSpatialConfigurationNet,
    ProbSpatialConfigurationNet,
    SpatialConfigurationNet,
)

__all__ = [
    "get_model",
    "Hourglass",
    "StackedHourglass",
    "CholeskyHourglass",
    "StackedCholeskyHourglass",
    "SpatialConfigurationNet",
    "OriginalSpatialConfigurationNet",
    "ProbSpatialConfigurationNet",
    "CoordConvLayer",
    "AddCoordChannels",
]


def get_model(
    model_name: str,
    path_to_model: Optional[str] = None,
    spatial_dims: int = 2,
    in_channels: int = 1,
    out_channels: int = 1,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> nn.Module:
    """
    Function to get a model, either pretrained or not.

    Args:
        model_name (str): name of the model to get.
        path_to_model (str, optional): path to the pretrained model. Defaults to None.
        spatial_dims (int, optional): number of spatial dimensions of the input image.
            Defaults to 2.
        in_channels (int, optional): number of input channels. Defaults to 1.
        out_channels (int, optional): number of output channels. Defaults to 1.
        device (torch.device, optional): device to use. Defaults to torch.device("cpu").
        **kwargs: additional keyword arguments to pass to the model.
    """
    if model_name == "SpatialConfigurationNet":
        model: nn.Module = SpatialConfigurationNet(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif model_name == "OriginalSpatialConfigurationNet":
        model = OriginalSpatialConfigurationNet(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif model_name == "ProbSpatialConfigurationNet":
        model = ProbSpatialConfigurationNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            sp_image_input=False,
            **kwargs,
        )
    elif model_name == "CoordConvProbSpatialConfigurationNet":
        model = nn.Sequential(
            CoordConvLayer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=64,
                radial_channel=True,
            ),
            ProbSpatialConfigurationNet(
                spatial_dims=spatial_dims,
                in_channels=64,
                out_channels=out_channels,
                sp_image_input=False,
                **kwargs,
            ),
        )
    elif model_name == "UNet":
        if not kwargs:
            model = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(128, 128, 128, 128, 128, 128),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                norm="instance",
            )
        else:
            model = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
    elif model_name == "CoordConvUNet":
        if not kwargs:
            model = nn.Sequential(
                CoordConvLayer(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=64,
                    radial_channel=True,
                ),
                UNet(
                    spatial_dims=spatial_dims,
                    in_channels=64,
                    out_channels=out_channels,
                    channels=(128, 128, 128, 128, 128),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                    norm="instance",
                ),
            )
        else:
            model = nn.Sequential(
                CoordConvLayer(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=64,
                    radial_channel=True,
                ),
                UNet(
                    spatial_dims=spatial_dims, in_channels=64, out_channels=out_channels, **kwargs
                ),
            )
    else:
        raise ValueError(f"Model name {model_name} not recognized.")

    if path_to_model is not None:
        print("Loading pretrained model...")
        if not os.path.isfile(path_to_model):
            print("Could not find model. Check if the path is correct.")
            return model
        print(path_to_model)
        try:
            model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))
        except RuntimeError:
            try:
                if device == torch.device("cpu"):
                    model.load_state_dict(
                        torch.load(path_to_model, map_location=torch.device("cuda"))
                    )
                    model.to(device)
                else:
                    model.load_state_dict(
                        torch.load(path_to_model, map_location=torch.device("cpu"))
                    )
                    model.to(device)
            except RuntimeError:
                print(
                    "Could not load model. Check if the model is compatible with the current "
                    "version of PyTorch."
                )

    return model
