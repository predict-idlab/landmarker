"""
Losses module for training landmark localization models.
"""

from .losses import (
    GeneralizedNormalHeatmapLoss,
    EuclideanDistanceJSDivergenceReg,
    EuclideanDistanceVarianceReg,
    MultivariateGaussianNLLLoss,
    WingLoss,
    AdaptiveWingLoss,
    StarLoss,
    StackedLoss,
    GaussianHeatmapL2Loss,
)

__all__ = [
    "GeneralizedNormalHeatmapLoss",
    "EuclideanDistanceJSDivergenceReg",
    "EuclideanDistanceVarianceReg",
    "MultivariateGaussianNLLLoss",
    "WingLoss",
    "AdaptiveWingLoss",
    "StarLoss",
    "StackedLoss",
    "GaussianHeatmapL2Loss",
]
