"""Pipeline module"""

from .pipeline import (
    AdaptiveHeatmapRegressionPipeline,
    CoordinateRegressionPipeline,
    IndirectUncertaintyAwareHeatmapRegressionPipeline,
    IndirectUncertaintyUnawareHeatmapRegressionPipeline,
    StaticHeatmapRegressionPipeline,
)

__all__ = [
    "StaticHeatmapRegressionPipeline",
    "AdaptiveHeatmapRegressionPipeline",
    "IndirectUncertaintyAwareHeatmapRegressionPipeline",
    "IndirectUncertaintyUnawareHeatmapRegressionPipeline",
    "CoordinateRegressionPipeline",
]
