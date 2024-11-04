"""Module for defining models for landmark detection."""

from .cholesky_hourglass import CholeskyHourglass, StackedCholeskyHourglass
from .coord_conv import AddCoordChannels, CoordConvLayer
from .hourglass import Hourglass, StackedHourglass
from .spatial_configuration_net import (
    OriginalSpatialConfigurationNet,
    OriginalSpatialConfigurationNet3d,
    ProbSpatialConfigurationNet,
    SpatialConfigurationNet,
)

__all__ = [
    "Hourglass",
    "StackedHourglass",
    "CholeskyHourglass",
    "StackedCholeskyHourglass",
    "SpatialConfigurationNet",
    "OriginalSpatialConfigurationNet",
    "OriginalSpatialConfigurationNet3d",
    "ProbSpatialConfigurationNet",
    "CoordConvLayer",
    "AddCoordChannels",
]
