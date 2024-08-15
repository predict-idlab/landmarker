"""
Losses module for training landmark localization models.
"""

from .losses import (
    AdaptiveWingLoss,
    EuclideanDistanceJSDivergenceReg,
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
    "EuclideanDistanceJSDivergenceReg",
    "EuclideanDistanceVarianceReg",
    "MultivariateGaussianNLLLoss",
    "WingLoss",
    "AdaptiveWingLoss",
    "StarLoss",
    "StackedLoss",
    "GaussianHeatmapL2Loss",
    "NLLLoss",
]
