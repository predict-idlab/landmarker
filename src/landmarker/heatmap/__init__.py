"""
Heatmap module.
"""

from .decoder import (
    coord_argmax,
    coord_cov_from_gaussian_ls,
    coord_local_soft_argmax,
    coord_weighted_spatial_mean,
    coord_weighted_spatial_mean_cov,
    cov_from_gaussian_ls,
    heatmap_to_coord,
    heatmap_to_coord_cov,
    heatmap_to_coord_enlarge,
    heatmap_to_multiple_coord,
    non_maximum_surpression,
    weighted_sample_cov,
    windowed_weigthed_sample_cov,
)
from .generator import GaussianHeatmapGenerator, LaplacianHeatmapGenerator

__all__ = [
    "LaplacianHeatmapGenerator",
    "GaussianHeatmapGenerator",
    "coord_argmax",
    "coord_local_soft_argmax",
    "coord_weighted_spatial_mean",
    "heatmap_to_coord",
    "heatmap_to_coord_enlarge",
    "coord_soft_argmax_cov",
    "coord_weighted_spatial_mean_cov",
    "heatmap_to_coord_cov",
    "coord_cov_from_gaussian_ls",
    "cov_from_gaussian_ls",
    "weighted_sample_cov",
    "windowed_weigthed_sample_cov",
    "heatmap_to_multiple_coord",
    "non_maximum_surpression",
    "non_maximum_surpression_local_soft_argmax",
    "softmax_2d",
]
