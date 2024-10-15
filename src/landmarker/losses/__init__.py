"""
Losses module for training landmark localization models.
"""

from .losses import (
    AdaptiveWingLoss,
    EuclideanDistanceVarianceReg,
    GaussianHeatmapL2Loss,
    GeneralizedNormalHeatmapLoss,
    MultivariateGaussianNLLLoss,
    NLLLoss,
    StackedLoss,
    StarLoss,
    WingLoss,
)

__all__ = [
    "GeneralizedNormalHeatmapLoss",
    "EuclideanDistanceVarianceReg",
    "MultivariateGaussianNLLLoss",
    "WingLoss",
    "AdaptiveWingLoss",
    "StarLoss",
    "StackedLoss",
    "GaussianHeatmapL2Loss",
    "NLLLoss",
]
